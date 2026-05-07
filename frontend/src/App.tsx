import React, { useState, useEffect, useRef } from 'react';
import { Upload, MessageSquare, ArrowUp, Zap, ServerIcon } from 'lucide-react';
import { marked } from 'marked';

interface Meeting {
  meeting_id: string;
}

interface Message {
  role: 'ai' | 'user';
  content: string;
}

export default function App() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [activeMeetingId, setActiveMeetingId] = useState<string | null>(null);
  const [activeIngestions, setActiveIngestions] = useState<Record<string, string>>({});
  
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', content: 'Hello! I am your Meeting Intelligence interface.\n\nSelect a meeting from the sidebar to perform a deep-dive analysis, or ask a general question to search across all meetings.' }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatLogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchMeetings();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (chatLogRef.current) {
        chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchMeetings = async () => {
    try {
      const res = await fetch('/api/meetings');
      const data = await res.json();
      setMeetings(data.meetings || []);
    } catch (e) {
      console.error(e);
    }
  };

  const checkStatus = async () => {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      setActiveIngestions(data.active || {});
    } catch (e) {
      console.error(e);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    
    setIsUploading(true);
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.meeting_id) {
        setActiveIngestions(prev => ({ ...prev, [data.meeting_id]: 'processing' }));
      }
    } catch (err) {
      alert("Upload failed.");
    }
    setIsUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const sendMessage = async () => {
    if (!query.trim() || isProcessing) return;

    const userQuery = query.trim();
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setIsProcessing(true);

    if (activeMeetingId) {
      // SSE Stream Logic
      setMessages(prev => [...prev, { role: 'ai', content: '' }]);
      
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userQuery, meeting_id: activeMeetingId, stream: true })
        });

        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");

        if (reader) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const dataStr = line.replace('data: ', '').trim();
                if (dataStr === '[DONE]') break;
                if (!dataStr) continue;

                try {
                  const parsed = JSON.parse(dataStr);
                  setMessages(prev => {
                    const newMessages = [...prev];
                    const lastMessage = newMessages[newMessages.length - 1];
                    lastMessage.content += parsed.token;
                    return newMessages;
                  });
                } catch (e) {
                  // JSON parse err safely ignored
                }
              }
            }
          }
        }
      } catch (err) {
        console.error("Stream failed", err);
      }
      setIsProcessing(false);
    } else {
      // Generic query
      setMessages(prev => [...prev, { role: 'ai', content: '*Searching global knowledge graph...*' }]);
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userQuery, stream: false })
        });
        const data = await res.json();
        
        setMessages(prev => {
          const newMessages = [...prev];
          let content = data.error || data.answer;
          if (data.sources && data.sources.length) {
              content += `\n\n<small style="color:#64748b">Sources: ${data.sources.join(', ')}</small>`;
          }
          newMessages[newMessages.length - 1].content = content;
          return newMessages;
        });
      } catch (err) {
        setMessages(prev => [...prev, { role: 'ai', content: 'Connection Error' }]);
      }
      setIsProcessing(false);
    }
  };

  // Combine fetched DB meetings + actively processing meetings 
  const allInjestions = [...meetings];
  for (const [id, status] of Object.entries(activeIngestions)) {
    if (status === 'processing' && !allInjestions.find(m => m.meeting_id === id)) {
      allInjestions.unshift({ meeting_id: id } as any);
    }
  }

  return (
    <div className="flex h-screen w-screen p-6 gap-6 box-border bg-slate-950 font-['Inter'] relative overflow-hidden">
      {/* Background Animated Gradients */}
      <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-indigo-600/10 blur-[120px] pointer-events-none mix-blend-screen animate-pulse duration-10000" />
      <div className="absolute -bottom-[20%] -right-[10%] w-[50%] h-[50%] rounded-full bg-purple-600/10 blur-[120px] pointer-events-none mix-blend-screen animate-pulse duration-7000" />

      {/* Sidebar */}
      <aside className="w-80 flex flex-col bg-slate-900/60 backdrop-blur-3xl border border-white/10 rounded-[32px] p-6 shadow-2xl z-20 relative">
        <div className="flex items-center justify-between mb-8 px-2">
          <h2 className="text-2xl font-bold font-['Outfit'] bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent tracking-tight">meetingLLM</h2>
          <div className="bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 px-3 py-1 rounded-full text-[0.65rem] font-bold tracking-widest uppercase shadow-inner">Vite</div>
        </div>

        <button 
          onClick={() => {
            setActiveMeetingId(null);
            setMessages([{ role: 'ai', content: 'Global Search Mode Enabled. You can search across all ingested audio.'}]);
          }}
          className="bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-2xl p-3.5 flex items-center justify-center gap-2 font-semibold shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] hover:-translate-y-1 transition-all duration-300 mb-4"
        >
          <MessageSquare size={18} /> New Analysis
        </button>

        <input type="file" ref={fileInputRef} onChange={handleUpload} accept="audio/*,video/*" className="hidden" />
        
        <button 
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="bg-transparent border border-white/10 text-slate-400 rounded-2xl p-3.5 flex items-center justify-center gap-2 font-semibold hover:bg-white/5 hover:text-white transition-all duration-300 mb-8"
        >
          <Upload size={18} /> {isUploading ? 'Uploading...' : 'Upload Audio'}
        </button>

        <div className="text-[0.65rem] text-slate-500 font-bold tracking-widest uppercase mb-4 pl-3">Available Meetings</div>
        
        <div className="flex-1 overflow-y-auto pr-2 space-y-1.5 custom-scrollbar">
          {allInjestions.length === 0 ? (
            <div className="text-slate-500 text-sm italic p-2 text-center mt-4">No meetings found.</div>
          ) : (
            allInjestions.map(mtg => {
              const isActive = activeMeetingId === mtg.meeting_id;
              const isProcessingMtg = (mtg as any).processing || activeIngestions[mtg.meeting_id] === 'processing';
              return (
                <button
                  key={mtg.meeting_id}
                  onClick={() => {
                    if (!isProcessingMtg) {
                      setActiveMeetingId(mtg.meeting_id);
                      setMessages([{ role: 'ai', content: `Preparing deep analysis for **${mtg.meeting_id}**.\n\nWhat would you like to know?`}]);
                    }
                  }}
                  disabled={isProcessingMtg}
                  className={`w-full text-left px-4 py-3.5 rounded-xl transition-all duration-300 flex justify-between items-center group relative overflow-hidden ${isActive ? 'bg-indigo-500/15 text-white font-medium border border-indigo-500/30' : 'text-slate-400 hover:text-white hover:bg-white/5 border border-transparent'}`}
                >
                  {isActive && <div className="absolute left-0 top-0 bottom-0 w-1 bg-indigo-400 rounded-full" />}
                  <span className="truncate pr-2 relative z-10">{mtg.meeting_id}</span>
                  {isProcessingMtg && <span className="bg-amber-500/15 text-amber-400 border border-amber-500/30 px-2 py-0.5 rounded shadow-[0_0_12px_rgba(245,158,11,0.2)] animate-pulse text-[0.6rem] font-bold uppercase tracking-wider relative z-10">Processing</span>}
                </button>
              );
            })
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative rounded-[32px] overflow-hidden bg-slate-900/40 backdrop-blur-xl border border-white/5 shadow-2xl z-10">
        {/* Header */}
        <header className="h-24 flex items-center px-10 z-10 bg-gradient-to-b from-slate-900/80 to-transparent border-b border-white/5">
          <h3 className="font-['Outfit'] font-semibold text-2xl text-white mr-4 tracking-tight drop-shadow-sm">
            {activeMeetingId ? `Deep Dive: ${activeMeetingId}` : 'Global Knowledge Graph'}
          </h3>
          <span className="px-3.5 py-1.5 rounded-full text-[0.75rem] font-bold tracking-widest uppercase bg-white/5 text-slate-400 border border-white/10 shadow-inner">
            {activeMeetingId ? 'Detail Mode' : 'Global Mode'}
          </span>
        </header>

        {/* Chat Log */}
        <div ref={chatLogRef} className="flex-1 overflow-y-auto px-6 md:px-12 pb-48 flex flex-col gap-6 pt-6 custom-scrollbar scroll-smooth">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-5 p-6 w-full max-w-4xl mx-auto rounded-3xl animate-slide-up transition-all ${msg.role === 'ai' ? 'bg-slate-800/50 backdrop-blur-md border border-white/5 shadow-2xl' : 'bg-transparent'}`}>
              <div className={`w-11 h-11 rounded-2xl flex items-center justify-center shrink-0 shadow-xl ${msg.role === 'user' ? 'bg-gradient-to-br from-slate-700 to-slate-800 border border-white/10' : 'bg-gradient-to-br from-indigo-500 to-purple-600'}`}>
                {msg.role === 'user' ? <Zap size={20} className="text-white"/> : <ServerIcon size={20} className="text-white"/>}
              </div>
              <div className="w-full pt-1 overflow-hidden">
                <div 
                  className={`markdown-content w-full ${msg.role === 'user' ? 'text-white text-[1.1rem] leading-relaxed font-medium' : 'text-slate-300 leading-relaxed'}`}
                  dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) }} 
                />
                {msg.role === 'ai' && i === messages.length - 1 && isProcessing && <span className="inline-block w-2.5 h-5 bg-indigo-400 rounded animate-pulse shadow-[0_0_12px_rgba(129,140,248,0.6)] ml-2 align-middle"></span>}
              </div>
            </div>
          ))}
        </div>

        {/* Input Box Generator Zone */}
        <div className="absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-slate-950 via-slate-950/80 to-transparent pointer-events-none z-20">
          <div className="max-w-4xl mx-auto bg-slate-800/90 backdrop-blur-2xl border border-white/10 rounded-[32px] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.8)] p-2 flex items-end pointer-events-auto transition-all duration-300 focus-within:border-indigo-400/50 focus-within:shadow-[0_0_0_1px_rgba(129,140,248,0.3),0_24px_60px_rgba(0,0,0,0.6)] focus-within:bg-slate-800">
            <textarea 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Message meetingLLM..."
              className="flex-1 bg-transparent border-none text-white p-4 pl-6 font-['Inter'] text-[1.05rem] resize-none outline-none max-h-48 placeholder-slate-500"
              rows={Math.min(5, query.split('\n').length)}
            />
            <button 
              onClick={sendMessage}
              disabled={isProcessing || !query.trim()}
              className="w-12 h-12 bg-white text-slate-900 rounded-full flex items-center justify-center shrink-0 ml-2 shadow-xl transition-all duration-300 hover:scale-[1.08] hover:bg-indigo-50 disabled:bg-white/10 disabled:text-white/30 disabled:scale-100 disabled:cursor-not-allowed mb-2 mr-2"
            >
              <ArrowUp size={22} strokeWidth={2.5}/>
            </button>
          </div>
          <div className="text-center text-[0.7rem] tracking-wide text-slate-500 font-['Outfit'] mt-5">
            Meeting Intelligence generates responses based on strict RAG chunk boundaries.
          </div>
        </div>
      </main>
    </div>
  );
}
