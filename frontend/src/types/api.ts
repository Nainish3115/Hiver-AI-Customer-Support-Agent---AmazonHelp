export interface SupportProcessRequest {
  customer_message: string;
  case_id?: string;
  policy_type?: string;
}

export interface SupportProcessResponse {
  case_id?: string;
  customer_message: string;
  predicted_intent: string;
  intent_confidence: number;
  decision: "AUTO_HANDLE" | "ESCALATE_TO_HUMAN" | string;
  reason_code: string;
  decision_reason: string;
  draft_response: string;
  risk_score: number;
  safety_check_passed: boolean;
  grounding_status: string;
  selected_evidence_count: number;
  retrieved_case_ids: string[];
  retrieved_scores: number[];
  signals: Record<string, any>;
  generation_metadata?: Record<string, any>;
  latency_ms: number;
  request_id?: string;
}

export interface HealthResponse {
  status: "ok" | "error";
  service: string;
  pipeline_ready: boolean;
  version: string;
  request_id?: string;
}

export interface ReadyResponse {
  status: "ready" | "not_ready";
  service: string;
  pipeline_ready: boolean;
  vector_store_documents: number;
  version: string;
  request_id?: string;
}

export interface MetricDetail {
  value: number;
  evaluation_set?: string;
  n_examples?: number;
  note?: string;
  max_score?: number;
  status?: string;
}

export interface StatsResponse {
  service: string;
  version: string;
  target_brand: string;
  canonical_corpus_size: number;
  raw_conversation_turns: number;
  intent_taxonomy_count: number;
  active_intent_classifier: string;
  active_retrieval_mode: string;
  active_llm_model: string;
  active_risk_threshold: number;
  metrics: Record<string, MetricDetail>;
  request_id?: string;
}

export interface RetrievalSearchResult {
  case_id: string;
  customer_problem: string;
  score: number;
  intent?: string;
}

export interface RetrievalSearchResponse {
  query: string;
  total_retrieved: number;
  results: RetrievalSearchResult[];
  latency_ms: number;
  request_id?: string;
}

export interface APIErrorDetail {
  code: string;
  message: string;
  detail?: string;
  request_id?: string;
}
