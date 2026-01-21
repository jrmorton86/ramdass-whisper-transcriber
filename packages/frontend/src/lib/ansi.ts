// ANSI color code parser for log rendering
const ANSI_COLORS: Record<string, string> = {
  '30': '#000000', // black
  '31': '#ef4444', // red
  '32': '#22c55e', // green
  '33': '#eab308', // yellow
  '34': '#3b82f6', // blue
  '35': '#a855f7', // magenta
  '36': '#06b6d4', // cyan
  '37': '#ffffff', // white
  '90': '#6b7280', // bright black (gray)
  '91': '#f87171', // bright red
  '92': '#4ade80', // bright green
  '93': '#fde047', // bright yellow
  '94': '#60a5fa', // bright blue
  '95': '#c084fc', // bright magenta
  '96': '#22d3ee', // bright cyan
  '97': '#f9fafb', // bright white
};

export interface ParsedAnsiSegment {
  text: string;
  color?: string;
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
}

export function parseAnsi(text: string): ParsedAnsiSegment[] {
  const segments: ParsedAnsiSegment[] = [];
  const regex = /\x1b\[([0-9;]+)m/g;

  let lastIndex = 0;
  let currentStyle: Partial<ParsedAnsiSegment> = {};
  let match;

  while ((match = regex.exec(text)) !== null) {
    // Add text before this escape sequence
    if (match.index > lastIndex) {
      const segmentText = text.substring(lastIndex, match.index);
      if (segmentText) {
        segments.push({ text: segmentText, ...currentStyle });
      }
    }

    // Parse the escape sequence
    const codes = match[1].split(';');
    for (const code of codes) {
      if (code === '0') {
        currentStyle = {}; // Reset
      } else if (code === '1') {
        currentStyle.bold = true;
      } else if (code === '3') {
        currentStyle.italic = true;
      } else if (code === '4') {
        currentStyle.underline = true;
      } else if (ANSI_COLORS[code]) {
        currentStyle.color = ANSI_COLORS[code];
      }
    }

    lastIndex = regex.lastIndex;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    const remainingText = text.substring(lastIndex);
    if (remainingText) {
      segments.push({ text: remainingText, ...currentStyle });
    }
  }

  return segments.length > 0 ? segments : [{ text }];
}

export function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]+m/g, '');
}
