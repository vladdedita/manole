import { useState } from "react";
import { motion } from "motion/react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { AgentStep } from "../hooks/useChat";

function truncateParams(params: Record<string, unknown>, maxLen = 80): string {
  const str = JSON.stringify(params);
  return str.length > maxLen ? str.slice(0, maxLen) + "\u2026" : str;
}

interface AgentStepsProps {
  steps: AgentStep[];
}

export function AgentSteps({ steps }: AgentStepsProps) {
  const [open, setOpen] = useState(false);

  if (steps.length === 0) return null;

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-2">
      <CollapsibleTrigger
        className="font-mono text-[10px] uppercase tracking-widest text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer"
      >
        {steps.length} step{steps.length !== 1 ? "s" : ""}
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="relative mt-2 ml-2 border-l border-accent/30 pl-4">
          {steps.map((step, i) => (
            <motion.div
              key={`step-${i}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08, duration: 0.2 }}
              className="relative mb-2 last:mb-0"
            >
              {/* Timeline dot */}
              <div className="absolute -left-[21px] top-1.5 h-2 w-2 rounded-full bg-accent" />

              <div className="flex flex-wrap items-baseline gap-2">
                <span className="font-mono text-[11px] text-accent bg-accent/10 px-1.5 py-0.5 rounded">
                  {step.tool}
                </span>
                <span className="font-mono text-[11px] text-text-tertiary leading-tight">
                  {truncateParams(step.params)}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
