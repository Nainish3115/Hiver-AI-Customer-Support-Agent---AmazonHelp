"""
Historical customer support case RAG corpus builder and document formatter.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


def extract_case_resolution(tweets_list: List[Dict[str, Any]]) -> Tuple[str, str, str, bool]:
    """
    Extract customer problem, support interaction, and resolution from tweet list.
    Returns:
      (customer_problem, support_interaction, resolution, has_resolution)
    """
    if not tweets_list:
        return "", "", "", False

    customer_tweets = [t for t in tweets_list if t.get('inbound')]
    support_tweets = [t for t in tweets_list if not t.get('inbound')]

    customer_problem = customer_tweets[0]['text'] if customer_tweets else ""

    interaction_lines = []
    for t in tweets_list:
        auth = t.get('author_id', 'unknown')
        txt = t.get('text', '')
        interaction_lines.append(f"[{auth}]: {txt}")

    support_interaction = "\n".join(interaction_lines)

    if not support_tweets:
        return customer_problem, support_interaction, "", False

    final_support_tweet = support_tweets[-1]['text']

    # Filter out pure DM redirection responses without actual resolution text
    text_lower = final_support_tweet.lower()
    is_pure_dm_redirection = (
        ("send us a dm" in text_lower or "please dm us" in text_lower or "shoot us a dm" in text_lower)
        and len(final_support_tweet) < 90
    )

    has_res = not is_pure_dm_redirection

    return customer_problem, support_interaction, final_support_tweet, has_res


def format_rag_document_text(intent: str, customer_problem: str, support_interaction: str, resolution: str) -> str:
    """
    Create a clean, deterministic text representation for each RAG document.
    """
    return (
        f"INTENT: {intent}\n"
        f"CUSTOMER PROBLEM: {customer_problem}\n"
        f"SUPPORT INTERACTION:\n{support_interaction}\n"
        f"RESOLUTION: {resolution}"
    )


def build_amazonhelp_rag_corpus(
    threads_parquet_path: str,
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build historical customer-support case corpus from eligible AmazonHelp conversations.
    """
    path = Path(threads_parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed threads parquet not found at {threads_parquet_path}")

    df_threads = pd.read_parquet(path)

    # Filter eligible conversations
    if 'is_rag_eligible' in df_threads.columns:
        df_eligible = df_threads[df_threads['is_rag_eligible']].copy()
    else:
        df_eligible = df_threads[df_threads['contains_both']].copy()

    total_eligible = len(df_eligible)

    cases = []
    documents_jsonl = []
    resolution_lengths = []
    seen_hashes = set()
    duplicate_count = 0
    with_resolution_cnt = 0

    for _, row in df_eligible.iterrows():
        cid = int(row['conversation_id'])
        intent_name = str(row.get('discovered_intent', 'general_inquiry_other'))
        tweets_json_str = row.get('tweets_json', '[]')

        try:
            tw_list = json.loads(tweets_json_str) if isinstance(tweets_json_str, str) else tweets_json_str
        except Exception:
            tw_list = []

        prob, interact, res, has_res = extract_case_resolution(tw_list)

        # Check deduplication hash of problem text
        prob_hash = hashlib.md5(prob.strip().lower().encode('utf-8')).hexdigest()
        is_dup = prob_hash in seen_hashes
        seen_hashes.add(prob_hash)

        if is_dup:
            duplicate_count += 1

        if has_res:
            with_resolution_cnt += 1
            resolution_lengths.append(len(res))

        doc_text = format_rag_document_text(intent_name, prob, interact, res)

        case_id = f"RAG-AMAZON-{cid}"

        case_dict = {
            "case_id": case_id,
            "conversation_id": cid,
            "intent": intent_name,
            "customer_problem": prob,
            "support_interaction": interact,
            "resolution": res,
            "full_conversation": interact,
            "timestamp": str(row.get('start_timestamp', '')),
            "conversation_length": int(row.get('num_tweets', 2)),
            "has_resolution": has_res,
            "is_duplicate": is_dup,
            "document_text": doc_text
        }

        cases.append(case_dict)

        # High-quality document payload for jsonl
        documents_jsonl.append({
            "case_id": case_id,
            "intent": intent_name,
            "document_text": doc_text,
            "metadata": {
                "conversation_id": cid,
                "timestamp": str(row.get('start_timestamp', '')),
                "conversation_length": int(row.get('num_tweets', 2)),
                "has_resolution": has_res
            }
        })

    df_rag = pd.DataFrame(cases)

    # Save to parquet and jsonl
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    parquet_out = out_p / "amazonhelp_rag_corpus.parquet"
    jsonl_out = out_p / "amazonhelp_rag_documents.jsonl"

    df_rag.to_parquet(parquet_out, index=False)

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for doc in documents_jsonl:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    stats = {
        "total_eligible_conversations": total_eligible,
        "conversations_with_resolution": with_resolution_cnt,
        "conversations_without_resolution": total_eligible - with_resolution_cnt,
        "resolution_rate_pct": round((with_resolution_cnt / total_eligible) * 100, 2) if total_eligible > 0 else 0.0,
        "avg_resolution_char_length": round(float(np.mean(resolution_lengths)), 2) if resolution_lengths else 0.0,
        "duplicate_problem_cases": duplicate_count,
        "unique_cases_count": total_eligible - duplicate_count
    }

    return df_rag, documents_jsonl, stats
