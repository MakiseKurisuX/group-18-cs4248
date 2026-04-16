import os
import pandas as pd
import instructor
from anthropic import Anthropic
from pydantic import BaseModel
from typing import List, Literal
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
INPUT_FILE = "data/output/testing_dataset_bom.csv"
OUTPUT_FILE = "data/output/testing_dataset_final.csv"
CHECKPOINT_FILE = "data/output/filtered_evaluations_checkpoint.csv"
BATCH_SIZE = 100
MODEL_NAME = "claude-haiku-4-5-20251001"

# Initialize Anthropic Client via Instructor
# Make sure to set your ANTHROPIC_API_KEY environment variable
client = instructor.from_anthropic(Anthropic())

# --- Pydantic Schema ---
class Evaluation(BaseModel):
    id: int
    verdict: Literal["KEEP", "DROP"]

class BatchEvaluations(BaseModel):
    evaluations: List[Evaluation]

# --- Prompting String ---
SYSTEM_PROMPT = """You are an expert linguistic data cleaner for a machine learning dataset. You will be given a list of satirical news headlines. Your job is to determine if the sarcasm can be detected purely through linguistic structure (KEEP) or if it requires outside human context to understand (DROP).

Output ONLY 'DROP' if the headline meets ANY of these conditions:
1. World Knowledge: The joke strictly requires knowing the history, reputation, or current events surrounding specific celebrities, politicians, brands, or sports teams to be funny.
2. Perfect Mimicry (No Absurdity): The text reads exactly like a boring, realistic, 100% normal news alert (e.g., 'senate passes new tax bill'). If there is no linguistic exaggeration, drop it.
3. Missing Visual Context: The text reads like a neutral caption for a photograph we cannot see (e.g., 'the new poster for the movie', 'look at this guy').
4. Cultural Irony: The humor relies on unspoken societal norms, lived experiences, or stereotypes rather than structural wordplay.

Output ONLY 'KEEP' if the headline contains clear linguistic signals of sarcasm:
1. Structural Absurdity: It uses extreme hyperbole, oxymorons, or physically impossible scenarios. (Note: The Absurdity Override - Even if a headline contains a real celebrity name, if the action they are doing is completely absurd or impossible, KEEP it).
2. Tone Clash: It pairs highly formal/academic language with ridiculous, gross, or mundane concepts.
3. The 'Area Man' Trope: It uses fake, generic placeholders (like 'Area Man', 'Nation's Dogs', 'Local Mom') in exaggerated, highly descriptive situations.

Output the results matching the requested JSON schema."""

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def evaluate_batch_with_retry(batch_text: str) -> BatchEvaluations:
    """Sends the formatted batch to Claude Haiku and extracts the structured JSON response."""
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": batch_text}
        ],
        response_model=BatchEvaluations
    )

def main():
    print(f"Loading dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    
    # We will generate our own simple sequentially increasing ID just for this process
    # This guarantees no weird index conflicts
    df["filter_id"] = range(1, len(df) + 1)
    
    total_rows = len(df)
    
    # Load checkpoint if it exists
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_df = pd.read_csv(CHECKPOINT_FILE, encoding='utf-8-sig')
        processed_ids = set(checkpoint_df["id"].tolist())
        print(f"Resuming from checkpoint. {len(processed_ids)} rows already processed.")
    else:
        processed_ids = set()
        # Create empty checkpoint file with headers
        pd.DataFrame(columns=["id", "verdict"]).to_csv(CHECKPOINT_FILE, index=False, encoding='utf-8-sig')
        print("Starting fresh. Created new checkpoint file.")

    # Filter down to only sarcastic rows that haven't been processed yet
    unprocessed_sarcastic_df = df[
        (~df["filter_id"].isin(processed_ids)) & 
        (df["is_sarcastic"] == 1)
    ]
    
    total_unprocessed = len(unprocessed_sarcastic_df)
    print(f"Found {total_unprocessed} remaining sarcastic rows to evaluate.")

    # Iterate purely over the sarcastic rows in chunks of BATCH_SIZE
    for start_idx in range(0, total_unprocessed, BATCH_SIZE):
        unprocessed_batch = unprocessed_sarcastic_df.iloc[start_idx:start_idx + BATCH_SIZE]
        
        print(f"Processing sarcastic batch: rows {start_idx + 1} to {min(start_idx + BATCH_SIZE, total_unprocessed)}...")
        
        # Build the numbered string payload
        batch_lines = []
        for _, row in unprocessed_batch.iterrows():
            batch_lines.append(f"{row['filter_id']}. {row['headline']}")
            
        batch_text = "\\n".join(batch_lines)
        
        try:
            # Call LLM
            print(f"Sending batch to {MODEL_NAME}...")
            response: BatchEvaluations = evaluate_batch_with_retry(batch_text)
            
            # Create a lookup dictionary
            id_to_headline = dict(zip(unprocessed_batch['filter_id'], unprocessed_batch['headline']))

            # Print evaluations with their original headlines
            print("\n--- LLM Verification Output ---")
            for eval_item in response.evaluations:
                headline = id_to_headline.get(eval_item.id, "(Headline Not Found)")
                print(f"[{eval_item.verdict}] {headline}")
            print("------------------------------------------\n")
            
            # Map results to dataframe and save immediately (checkpointing)
            batch_results = []
            for eval_item in response.evaluations:
                batch_results.append({"id": eval_item.id, "verdict": eval_item.verdict})
                
            batch_results_df = pd.DataFrame(batch_results)
            batch_results_df.to_csv(CHECKPOINT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
            
            print(f"Successfully processed and saved {len(batch_results)} evaluations.")
            
            # Small delay to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            import traceback
            print(f"Failed to process batch ending at {start_idx + BATCH_SIZE} after retries. Error trace:")
            print(traceback.format_exc())
            print("Stopping execution. You can restart the script to resume from the last checkpoint.")
            break
            
    print("Done calling API. Merging results back to main dataset...")
    
    # Final merge
    if os.path.exists(CHECKPOINT_FILE):
        final_evals = pd.read_csv(CHECKPOINT_FILE, encoding='utf-8-sig')
        # Merge back on our generated filter_id
        final_df = df.merge(final_evals, left_on="filter_id", right_on="id", how="left")
        
        # Determine how many we kept (Keep if verdict is KEEP OR if it wasn't evaluated)
        kept_df = final_df[(final_df["verdict"] == "KEEP") | (final_df["verdict"].isna())].copy()
        
        # Clean up temporary columns
        kept_df.drop(columns=["filter_id", "id", "verdict"], inplace=True)
        
        kept_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\\nFiltering Complete!")
        print(f"Original Dataset Size: {total_rows}")
        print(f"Filtered Dataset Size: {len(kept_df)} (A {(total_rows - len(kept_df))/total_rows*100:.1f}% reduction)")
        print(f"Saved directly to: {OUTPUT_FILE}")
    
if __name__ == "__main__":
    main()
