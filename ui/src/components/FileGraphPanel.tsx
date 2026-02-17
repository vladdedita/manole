import { useState, useCallback, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  ReactFlow,
  Background,
  MiniMap,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeTypes,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "@dagrejs/dagre";
import { FileGraphNode } from "./FileGraphNode";
import type { FileGraphData, FileEdge } from "../lib/protocol";

type TabId = "similarity" | "reference" | "structure";

const TABS: { id: TabId; label: string }[] = [
  { id: "similarity", label: "Similarity" },
  { id: "reference", label: "References" },
  { id: "structure", label: "Structure" },
];

const nodeTypes: NodeTypes = {
  file: FileGraphNode,
};

const FILE_TYPE_COLORS: Record<string, string> = {
  pdf: "#c9943e",
  md: "#6aad6a",
  txt: "#a59888",
  py: "#7aa2d4",
  js: "#d4a85c",
  ts: "#d4a85c",
  dir: "#635a50",
};

function getEdgeStyle(edge: FileEdge, tab: TabId): Partial<Edge> {
  if (tab === "similarity") {
    return {
      style: {
        stroke: "#c9943e",
        strokeOpacity: 0.15 + edge.weight * 0.45,
        strokeWidth: 1.5,
      },
      type: "default",
    };
  }
  if (tab === "reference") {
    return {
      style: { stroke: "#a59888", strokeWidth: 1.5 },
      type: "straight",
      markerEnd: { type: MarkerType.ArrowClosed, color: "#a59888", width: 12, height: 12 },
      label: edge.label,
      labelStyle: { fontSize: 9, fill: "#635a50" },
    };
  }
  // structure
  return {
    style: { stroke: "#2a2624", strokeWidth: 1, strokeDasharray: "4 4" },
    type: "smoothstep",
  };
}

function layoutNodes(
  nodes: Node[],
  edges: Edge[],
  tab: TabId,
): Node[] {
  if (nodes.length === 0) return nodes;

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));

  const isTree = tab === "structure";
  g.setGraph({
    rankdir: isTree ? "TB" : "LR",
    nodesep: isTree ? 60 : 100,
    ranksep: isTree ? 80 : 150,
    edgesep: 40,
  });

  for (const node of nodes) {
    g.setNode(node.id, { width: 140, height: 70 });
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: { x: pos.x - 70, y: pos.y - 35 },
    };
  });
}

interface FileGraphPanelProps {
  graph: FileGraphData | null;
  isLoading: boolean;
  error: string | null;
  onFetchGraph: () => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function FileGraphPanel({ graph, isLoading, error, onFetchGraph }: FileGraphPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>("similarity");
  const [threshold, setThreshold] = useState(0.6);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Fetch graph on mount if not loaded
  useEffect(() => {
    if (!graph && !isLoading && !error) {
      onFetchGraph();
    }
  }, [graph, isLoading, error, onFetchGraph]);

  // Filter edges by active tab and threshold
  const filteredEdges = useMemo(() => {
    if (!graph) return [];
    return graph.edges.filter((e) => {
      if (e.type !== activeTab) return false;
      if (activeTab === "similarity" && e.weight < threshold) return false;
      return true;
    });
  }, [graph, activeTab, threshold]);

  // Convert graph data to React Flow format
  useEffect(() => {
    if (!graph) return;

    const connectedNodeIds = new Set<string>();
    for (const e of filteredEdges) {
      connectedNodeIds.add(e.source);
      connectedNodeIds.add(e.target);
    }

    const rfNodes: Node[] = graph.nodes
      .filter((n) => connectedNodeIds.has(n.id))
      .map((n) => ({
        id: n.id,
        type: "file",
        position: { x: 0, y: 0 },
        data: {
          name: n.name,
          type: n.type,
          passageCount: n.passageCount,
          dir: n.dir,
          size: n.size,
          selected: n.id === selectedNodeId,
        },
      }));

    const rfEdges: Edge[] = filteredEdges
      .filter((e) => connectedNodeIds.has(e.source) && connectedNodeIds.has(e.target))
      .map((e, i) => ({
        id: `${e.source}-${e.target}-${i}`,
        source: e.source,
        target: e.target,
        ...getEdgeStyle(e, activeTab),
      }));

    const laidOut = layoutNodes(rfNodes, rfEdges, activeTab);
    setNodes(laidOut);
    setEdges(rfEdges);
  }, [graph, filteredEdges, activeTab, selectedNodeId, setNodes, setEdges]);

  const selectedNode = graph?.nodes.find((n) => n.id === selectedNodeId);
  const selectedEdgeCount = selectedNodeId
    ? filteredEdges.filter((e) => e.source === selectedNodeId || e.target === selectedNodeId).length
    : 0;

  const handleNodeClick = useCallback((_: unknown, node: Node) => {
    setSelectedNodeId((prev) => (prev === node.id ? null : node.id));
  }, []);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="inline-block h-2 w-2 rounded-full bg-warning"
          />
          <p className="mt-3 font-display text-lg italic text-text-tertiary">
            Building map...
          </p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="mx-4 px-4 py-2 rounded-lg border border-accent/40 bg-accent/10 text-accent text-sm font-sans">
          {error}
        </div>
      </div>
    );
  }

  // Empty state
  if (graph && graph.nodes.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <div className="font-display text-lg text-text-tertiary italic">No passages indexed</div>
          <p className="mt-1 font-sans text-xs text-text-tertiary/60">
            Index a folder to see its file graph
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Tab bar */}
      <div className="flex items-center gap-4 px-5 h-9 border-b border-border shrink-0">
        {TABS.map((tab) => {
          const count = graph?.edges.filter((e) => e.type === tab.id).length ?? 0;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative font-mono text-[10px] uppercase tracking-widest transition-colors pb-2 ${
                activeTab === tab.id
                  ? "text-accent"
                  : "text-text-tertiary hover:text-text-secondary"
              }`}
            >
              {tab.label}
              {count > 0 && (
                <span className="ml-1 text-text-tertiary">{count}</span>
              )}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="graph-tab-indicator"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-accent"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* React Flow canvas */}
      <div className="flex-1 min-h-0">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
          style={{ background: "var(--color-bg-primary)" }}
        >
          <Background color="var(--color-border)" gap={24} size={1} />
          <Controls
            showInteractive={false}
            className="!bg-bg-secondary !border-border !rounded-lg !shadow-none [&>button]:!bg-bg-secondary [&>button]:!border-border [&>button]:!text-text-secondary [&>button:hover]:!bg-bg-elevated"
          />
          <MiniMap
            nodeColor={(node) => {
              const type = (node.data as Record<string, unknown>)?.type as string;
              return FILE_TYPE_COLORS[type] || "#635a50";
            }}
            maskColor="rgba(20, 18, 16, 0.85)"
            className="!bg-bg-secondary !border !border-border !rounded-lg"
            style={{ opacity: 0.6 }}
          />
        </ReactFlow>
      </div>

      {/* Bottom tray */}
      <AnimatePresence mode="wait">
        <motion.div
          key={selectedNodeId ?? "summary"}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ duration: 0.15 }}
          className="border-t border-border bg-bg-secondary px-5 shrink-0"
        >
          {selectedNode ? (
            <div className="py-3 flex items-center gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-sans text-sm font-medium text-text-primary truncate">
                    {selectedNode.name}
                  </span>
                  <span className="font-mono text-[10px] text-text-tertiary">
                    {formatSize(selectedNode.size)}
                  </span>
                </div>
                <div className="mt-0.5 font-mono text-[10px] text-text-tertiary truncate">
                  {selectedNode.id}
                </div>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <span className="font-mono text-[10px] text-text-tertiary">
                  {selectedNode.passageCount} passages
                </span>
                <span className="font-mono text-[10px] text-text-tertiary">
                  {selectedEdgeCount} connections
                </span>
              </div>
            </div>
          ) : (
            <div className="py-2.5 flex items-center justify-between">
              <span className="font-mono text-[10px] text-text-tertiary">
                {nodes.length} files &middot; {edges.length} edges
              </span>
              {activeTab === "similarity" && (
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-text-tertiary">
                    Threshold: {threshold.toFixed(2)}
                  </span>
                  <input
                    type="range"
                    min="0.4"
                    max="1.0"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-24 h-1 rounded-full appearance-none bg-bg-elevated [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                </div>
              )}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
