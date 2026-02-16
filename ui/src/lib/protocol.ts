export interface Request {
  id: number;
  method: string;
  params: Record<string, unknown>;
}

export type ResponseType = "result" | "token" | "agent_step" | "error" | "status" | "progress" | "log";

export interface Response {
  id: number | null;
  type: ResponseType;
  data: Record<string, unknown>;
}

export interface TokenData {
  text: string;
}

export interface AgentStepData {
  step: number;
  tool: string;
  params: Record<string, unknown>;
}

export interface StatusData {
  state: "loading_model" | "indexing" | "ready" | "not_initialized";
}

export interface ProgressData {
  stage: string;
  percent: number;
}

export interface ResultData {
  text?: string;
  status?: string;
  indexName?: string;
  indexes?: string[];
  debug?: boolean;
}

export interface ErrorData {
  message: string;
}
