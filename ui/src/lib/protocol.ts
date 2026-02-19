export interface Request {
  id: number;
  method: string;
  params: Record<string, unknown>;
}

export type ResponseType = "result" | "token" | "agent_step" | "error" | "status" | "progress" | "log" | "directory_update" | "file_graph" | "setup_progress";

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
  state: "loading_model" | "indexing" | "summarizing" | "captioning" | "ready" | "not_initialized";
}

export interface ProgressData {
  stage: string;
  percent: number;
}

export interface ResultData {
  text?: string;
  sources?: string[];
  status?: string;
  indexName?: string;
  indexes?: string[];
  debug?: boolean;
}

export interface ErrorData {
  message: string;
}

export interface DirectoryStats {
  fileCount: number;
  totalSize: number;
  types: Record<string, number>;
  sizeByType: Record<string, number>;
  largestFiles: { name: string; size: number }[];
  avgFileSize: number;
  dirs: { count: number; maxDepth: number };
}

export interface DirectoryUpdateData {
  directoryId: string;
  state: "indexing" | "ready" | "error";
  stats?: DirectoryStats;
  summary?: string;
  error?: string;
}

export interface FileNode {
  id: string;
  name: string;
  type: string;
  size: number;
  dir: string;
  passageCount: number;
}

export interface FileEdge {
  source: string;
  target: string;
  type: "similarity" | "reference" | "structure";
  weight: number;
  label?: string;
}

export interface FileGraphData {
  nodes: FileNode[];
  edges: FileEdge[];
}

export interface SetupProgressData {
  model_id: string;
  filename?: string;
  downloaded_bytes?: number;
  total_bytes?: number;
  status: "downloading" | "verifying" | "complete" | "error" | "skipped";
  verified?: boolean;
  error?: string;
}
