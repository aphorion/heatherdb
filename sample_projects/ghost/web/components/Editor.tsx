"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { api, type Prediction } from "@/lib/api";

export default function Editor() {
  const [text, setText] = useState("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [stats, setStats] = useState({
    vocab_size: 0,
    patterns: 0,
    words: 0,
  });
  const [feeding, setFeeding] = useState(false);
  const [ghost, setGhost] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lastProcessedRef = useRef(0);

  // extract the last N words from text
  const getContext = useCallback(
    (t: string, n: number = 3): string[] => {
      const words = t.trim().split(/\s+/).filter(Boolean);
      return words.slice(-n);
    },
    []
  );

  // process new words as user types
  const processText = useCallback(
    async (newText: string) => {
      const words = newText.trim().split(/\s+/).filter(Boolean);
      const prevWords = text.trim().split(/\s+/).filter(Boolean);

      // detect if user just completed a word (typed a space)
      if (!newText.endsWith(" ") && !newText.endsWith("\n")) {
        setText(newText);
        setGhost("");
        return;
      }

      const currentWord = words[words.length - 1];
      if (!currentWord || words.length <= lastProcessedRef.current) {
        setText(newText);
        return;
      }

      setText(newText);
      lastProcessedRef.current = words.length;

      // send the word with context
      const context = words.slice(-4, -1); // last 3 words before current
      try {
        const res = await api.type(currentWord, context);
        setPredictions(res.predictions);
        setStats(res.stats);
        if (res.predictions.length > 0) {
          setGhost(res.predictions[0].word);
        } else {
          setGhost("");
        }
      } catch {
        // heather might not be running
      }
    },
    [text]
  );

  // handle Tab to accept prediction
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Tab" && ghost) {
        e.preventDefault();
        const newText = text + ghost + " ";
        setText(newText);
        setGhost("");
        lastProcessedRef.current += 1;

        // process the accepted word
        const words = newText.trim().split(/\s+/).filter(Boolean);
        const context = words.slice(-4, -1);
        api
          .type(ghost, context)
          .then((res) => {
            setPredictions(res.predictions);
            setStats(res.stats);
            if (res.predictions.length > 0) {
              setGhost(res.predictions[0].word);
            } else {
              setGhost("");
            }
          })
          .catch(() => {});
      }
    },
    [ghost, text]
  );

  // accept a clicked prediction
  const acceptPrediction = useCallback(
    (word: string) => {
      const newText = text + word + " ";
      setText(newText);
      setGhost("");
      lastProcessedRef.current += 1;
      textareaRef.current?.focus();

      const words = newText.trim().split(/\s+/).filter(Boolean);
      const context = words.slice(-4, -1);
      api
        .type(word, context)
        .then((res) => {
          setPredictions(res.predictions);
          setStats(res.stats);
          if (res.predictions.length > 0) {
            setGhost(res.predictions[0].word);
          } else {
            setGhost("");
          }
        })
        .catch(() => {});
    },
    [text]
  );

  // bulk feed text (paste or seed)
  const handleFeed = useCallback(async (feedText: string) => {
    if (!feedText.trim()) return;
    setFeeding(true);
    try {
      const res = await api.feed(feedText);
      setStats(res.stats);
      setPredictions(res.predictions);
      if (res.predictions.length > 0) {
        setGhost(res.predictions[0].word);
      }
      const words = feedText.trim().split(/\s+/).filter(Boolean);
      lastProcessedRef.current = words.length;
    } catch {
      // ignore
    } finally {
      setFeeding(false);
    }
  }, []);

  // handle paste — feed the pasted text to SDM
  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const pasted = e.clipboardData.getData("text");
      if (pasted.split(/\s+/).length > 3) {
        // bulk feed the pasted text
        const newText = text + pasted;
        setText(newText);
        handleFeed(newText);
        e.preventDefault();
      }
    },
    [text, handleFeed]
  );

  const handleReset = useCallback(async () => {
    await api.reset();
    setText("");
    setPredictions([]);
    setStats({ vocab_size: 0, patterns: 0, words: 0 });
    setGhost("");
    lastProcessedRef.current = 0;
    textareaRef.current?.focus();
  }, []);

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="flex flex-col h-full">
      {/* editor */}
      <div className="relative flex-1 min-h-0">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => processText(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder="Start typing. The EAM learns your patterns as you write..."
          className="w-full h-full min-h-[400px] bg-transparent text-zinc-100 text-lg leading-relaxed font-[family-name:var(--font-geist-mono)] resize-none focus:outline-none placeholder:text-zinc-700 p-0"
          autoFocus
          spellCheck={false}
        />
        {ghost && text.endsWith(" ") && (
          <div className="pointer-events-none absolute bottom-4 left-0">
            <span className="text-zinc-600 text-sm font-[family-name:var(--font-geist-mono)]">
              ghost: <span className="text-zinc-500">{ghost}</span>
              <span className="text-zinc-700 ml-2">Tab ↵</span>
            </span>
          </div>
        )}
      </div>

      {/* predictions */}
      {predictions.length > 0 && (
        <div className="flex items-center gap-2 py-4 border-t border-zinc-800/50">
          <span className="text-xs text-zinc-600 shrink-0">next:</span>
          {predictions.slice(0, 5).map((p, i) => (
            <button
              key={`${p.word}-${i}`}
              onClick={() => acceptPrediction(p.word)}
              className={`px-3 py-1.5 rounded-md text-sm font-[family-name:var(--font-geist-mono)] transition-all ${
                i === 0
                  ? "bg-zinc-800 text-zinc-200 border border-zinc-700"
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
              }`}
            >
              {p.word}
              <span className="ml-1.5 text-xs text-zinc-600">
                {Math.round(p.confidence * 100)}%
              </span>
            </button>
          ))}
        </div>
      )}

      {/* stats bar */}
      <div className="flex items-center justify-between py-3 border-t border-zinc-800/50 text-xs text-zinc-600 font-[family-name:var(--font-geist-mono)]">
        <div className="flex gap-4">
          <span>{wordCount} words typed</span>
          <span>{stats.vocab_size} vocabulary</span>
          <span>{stats.patterns} patterns learned</span>
        </div>
        <div className="flex items-center gap-3">
          {feeding && (
            <span className="text-violet-400 animate-pulse">learning...</span>
          )}
          {stats.patterns > 0 && (
            <button
              onClick={handleReset}
              className="text-zinc-700 hover:text-rose-400 transition-colors"
            >
              reset
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
