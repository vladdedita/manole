import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

const FILE_TYPE_COLORS: Record<string, string> = {
  pdf: "#c9943e",
  md: "#6aad6a",
  txt: "#a59888",
  py: "#7aa2d4",
  js: "#d4a85c",
  ts: "#d4a85c",
  json: "#a59888",
  csv: "#7aa2d4",
  dir: "#635a50",
};

function getTypeColor(type: string): string {
  return FILE_TYPE_COLORS[type] || "#635a50";
}

interface FileNodeData {
  name: string;
  type: string;
  passageCount: number;
  dir: string;
  size: number;
  selected?: boolean;
  [key: string]: unknown;
}

function FileGraphNodeComponent({ data, selected }: NodeProps) {
  const { name, type, passageCount } = data as unknown as FileNodeData;
  const color = getTypeColor(type as string);
  const isSelected = selected || (data as unknown as FileNodeData).selected;

  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-transparent !border-0 !w-2 !h-2" />
      <div
        className={`
          px-3 py-2 rounded-lg border transition-all duration-200
          bg-bg-elevated font-sans text-xs
          ${isSelected
            ? "border-accent/50 bg-accent/[0.06] shadow-[0_0_16px_rgba(201,148,62,0.12)]"
            : "border-border hover:border-accent/30 hover:shadow-[0_0_12px_rgba(201,148,62,0.08)]"
          }
        `}
        style={{ minWidth: 80, maxWidth: 160 }}
      >
        {type && (
          <span
            className="inline-block px-1.5 py-0.5 rounded font-mono text-[9px] uppercase mb-1"
            style={{
              backgroundColor: `${color}20`,
              color: color,
            }}
          >
            {type}
          </span>
        )}

        <div className="text-text-primary truncate leading-tight" title={name as string}>
          {name}
        </div>

        <div className="mt-1 font-mono text-[9px] text-text-tertiary">
          {passageCount} {Number(passageCount) === 1 ? "passage" : "passages"}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-transparent !border-0 !w-2 !h-2" />
    </>
  );
}

export const FileGraphNode = memo(FileGraphNodeComponent);
