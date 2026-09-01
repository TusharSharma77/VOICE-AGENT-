import React, { useState, useEffect, useRef } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export default function App() {
  const [urlInput, setUrlInput] = useState('');
  const [language, setLanguage] = useState('english');
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  // Active meeting data
  const [meetingData, setMeetingData] = useState(null);
  const [activeTab, setActiveTab] = useState('summary');

  // AI Assistant Chat state
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: "System ready. Paste a YouTube link on the left to analyze, or ask any question about indexed meetings."
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [serverOnline, setServerOnline] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetch(API_BASE + '/api/status')
      .then((res) => res.json())
      .then(() => setServerOnline(true))
      .catch(() => setServerOnline(false));

    fetchExistingDocs();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAsking]);

  const fetchExistingDocs = async () => {
    try {
      const res = await fetch(API_BASE + '/api/documents');
      if (res.ok) {
        const docs = await res.json();
        if (docs && docs.length > 0) {
          setMeetingData(docs[0]);
        }
      }
    } catch (e) {
      console.warn('Could not fetch documents:', e);
    }
  };

  const handleProcess = async (e) => {
    e.preventDefault();
    if (!urlInput.trim()) return;

    const source = urlInput.trim();
    setIsProcessing(true);
    setErrorMessage('');
    setStatusMessage('Downloading audio, transcribing, and compiling notes...');

    try {
      const res = await fetch(API_BASE + '/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: source, language })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || ('Processing failed with status ' + res.status));
      }

      const data = await res.json();
      setMeetingData(data);
      setActiveTab('summary');
      setStatusMessage('');
      setUrlInput('');

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: 'bot',
          text: 'Indexed "' + (data.title || 'Video Transcript') + '". You can now ask questions about this recording.'
        }
      ]);
    } catch (err) {
      setErrorMessage(err.message || 'An error occurred during processing.');
      setStatusMessage('');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendQuestion = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const q = chatInput.trim();
    setChatInput('');

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        sender: 'user',
        text: q
      }
    ]);

    setIsAsking(true);

    try {
      const res = await fetch(API_BASE + '/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to retrieve answer');
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: 'bot',
          text: data.answer,
          sourceDoc: meetingData?.title || 'Meeting Transcript'
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: 'bot',
          text: 'Error querying transcript: ' + err.message
        }
      ]);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="window-container">
      {/* Top macOS Style Window Header */}
      <div className="window-header">
        <div className="window-dots">
          <span className="dot dot-red"></span>
          <span className="dot dot-yellow"></span>
          <span className="dot dot-green"></span>
        </div>
        <div className="window-title">Video Agent // Meeting Intelligence</div>
        <div className="window-status">
          <span className={`status-dot ${serverOnline ? '' : 'offline'}`}></span>
          {serverOnline ? 'Backend Online' : 'Offline'}
        </div>
      </div>

      {/* Main 2-Panel Interior */}
      <div className="window-content">
        {/* Left Panel: Media Acquisition & Synthesized Notes */}
        <section className="panel-left">
          {/* Ingestion Box */}
          <div className="acquisition-box">
            <div className="box-label">Media Acquisition</div>
            <form onSubmit={handleProcess} className="acquisition-form">
              <input
                type="text"
                className="url-input"
                placeholder="Paste YouTube link (https://youtu.be/...) or local file path"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                disabled={isProcessing}
              />
              <select
                className="lang-dropdown"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={isProcessing}
              >
                <option value="english">English (Whisper)</option>
                <option value="hinglish">Hinglish (Sarvam AI)</option>
              </select>
              <button
                type="submit"
                className="btn-process"
                disabled={isProcessing || !urlInput.trim()}
              >
                {isProcessing ? 'Processing...' : 'Process'}
              </button>
            </form>

            {statusMessage && <div className="progress-banner">{statusMessage}</div>}
            {errorMessage && <div className="progress-banner" style={{ color: '#dc2626' }}>{errorMessage}</div>}
          </div>

          {/* Notes & Insights Output */}
          <div className="notes-card">
            {meetingData ? (
              <>
                <div className="notes-header">
                  <div className="notes-title">{meetingData.title || 'Processed Video'}</div>
                  <span className="badge-indexed">Indexed</span>
                </div>

                <div className="notes-tabs">
                  <button
                    className={`tab-btn ${activeTab === 'summary' ? 'active' : ''}`}
                    onClick={() => setActiveTab('summary')}
                  >
                    Summary
                  </button>
                  <button
                    className={`tab-btn ${activeTab === 'action_items' ? 'active' : ''}`}
                    onClick={() => setActiveTab('action_items')}
                  >
                    Action Items
                  </button>
                  <button
                    className={`tab-btn ${activeTab === 'key_decisions' ? 'active' : ''}`}
                    onClick={() => setActiveTab('key_decisions')}
                  >
                    Key Decisions
                  </button>
                  <button
                    className={`tab-btn ${activeTab === 'questions' ? 'active' : ''}`}
                    onClick={() => setActiveTab('questions')}
                  >
                    Questions
                  </button>
                  <button
                    className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`}
                    onClick={() => setActiveTab('transcript')}
                  >
                    Transcript
                  </button>
                </div>

                <div className="notes-content">
                  {activeTab === 'summary' && (meetingData.summary || 'No summary available.')}
                  {activeTab === 'action_items' && (meetingData.action_items || 'No action items found.')}
                  {activeTab === 'key_decisions' && (meetingData.key_decisions || 'No key decisions recorded.')}
                  {activeTab === 'questions' && (meetingData.questions || 'No open questions found.')}
                  {activeTab === 'transcript' && (meetingData.transcript || 'Transcript empty.')}
                </div>
              </>
            ) : (
              <div className="empty-view">
                <div>No media processed yet.</div>
                <div style={{ marginTop: '4px', color: 'var(--text-muted)' }}>
                  Paste a YouTube URL above and click Process to generate notes and index.
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Right Panel: AI Assistant (RAG Chat) */}
        <section className="panel-right">
          <div className="chat-top">
            <div className="chat-brand">
              <span className="status-dot"></span>
              AI Assistant (RAG)
            </div>
            <button
              className="btn-clear"
              onClick={() =>
                setMessages([
                  {
                    id: Date.now(),
                    sender: 'bot',
                    text: 'Chat cleared. Ask any question regarding the active transcript.'
                  }
                ])
              }
            >
              Clear
            </button>
          </div>

          <div className="chat-history">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`msg-row ${msg.sender === 'user' ? 'user' : 'bot'}`}
              >
                <div className={msg.sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-bot'}>
                  {msg.sourceDoc && (
                    <div className="citation-tag">Context: {msg.sourceDoc}</div>
                  )}
                  {msg.text}
                </div>
              </div>
            ))}

            {isAsking && (
              <div className="msg-row bot">
                <div className="chat-bubble-bot" style={{ color: 'var(--text-muted)' }}>
                  Retrieving context and answering...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-bottom">
            <form onSubmit={handleSendQuestion} className="chat-form">
              <input
                type="text"
                className="query-input"
                placeholder="Ask a question about the video..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={isAsking}
              />
              <button
                type="submit"
                className="btn-submit-chat"
                disabled={isAsking || !chatInput.trim()}
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </form>
            <div className="chat-note">
              Answers retrieved directly from transcript context.
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
