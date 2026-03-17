import Editor from "@/components/Editor";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="px-8 pt-8 pb-4">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-lg font-bold tracking-tight">Ghost</h1>
          <p className="text-xs text-zinc-600 mt-0.5">
            Type freely. The EAM learns your patterns and predicts your next
            word. No neural network — pure memory interference.
          </p>
        </div>
      </header>

      <main className="flex-1 px-8 pb-8">
        <div className="max-w-3xl mx-auto h-full">
          <Editor />
        </div>
      </main>
    </div>
  );
}
