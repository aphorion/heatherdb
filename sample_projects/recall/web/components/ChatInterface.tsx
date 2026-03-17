"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { api, type Message, type ChatResponse } from "@/lib/api";

interface ChatInterfaceProps {
  onResponse?: (res: ChatResponse) => void;
}

export default function ChatInterface({ onResponse }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.messages().then((d) => setMessages(d.messages)).catch(() => {});
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setInput("");
    setLoading(true);

    // optimistic user message
    setMessages((prev) => [
      ...prev,
      { id: -1, role: "user", content: text, timestamp: new Date().toISOString() },
    ]);

    try {
      const res = await api.chat(text);
      setMessages((prev) => [
        ...prev,
        {
          id: -2,
          role: "assistant",
          content: res.response,
          timestamp: new Date().toISOString(),
        },
      ]);
      onResponse?.(res);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: -3,
          role: "assistant",
          content: "Error: could not get response.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, onResponse]);

  return (
    <div className="flex-1 flex flex-col min-w-0">
      {/* messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && !loading && (
          <div className="text-center py-20">
            <p className="text-zinc-600 text-sm">
              Start a conversation. The EAM remembers everything.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={`${msg.id}-${i}`}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              data-testid={`message-${msg.role}`}
              className={`max-w-2xl rounded-lg px-4 py-3 ${
                msg.role === "user"
                  ? "bg-zinc-800 text-zinc-100"
                  : "bg-zinc-900 border border-zinc-800 text-zinc-200"
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3">
              <p className="text-sm text-zinc-500 animate-pulse">thinking...</p>
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      {/* input */}
      <div className="p-4 border-t border-zinc-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Type a message..."
            className="flex-1 px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="px-6 py-3 bg-zinc-100 text-zinc-900 rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
