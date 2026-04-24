import { useState } from 'react'
import './App.css'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

function App() {
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState([])
  const [answer, setAnswer] = useState("")
  const [loading, setLoading] = useState(false)
  const [collection] = useState("default")
  const [error, setError] = useState("")

  const uploadPDF = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    const formData = new FormData()
    formData.append("file", file)
    setError("")
    try {
      await axios.post(`${API}/ingest/pdf?collection_name=${collection}`, formData)
      alert(`${file.name} uploaded successfully`)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Upload failed")
    }
  }

  const askQuestion = async () => {
    if (!question.trim() || loading) return
    setLoading(true)
    setError("")
    const userMessage = { role: "user", content: question }
    const nextMessages = [...messages, userMessage]
    setMessages(nextMessages)
    setQuestion("")
    setAnswer("")

    // Build [[user, assistant], ...] tuples from prior messages only
    const chat_history = []
    for (let i = 0; i < messages.length - 1; i += 2) {
      if (messages[i]?.role === "user" && messages[i + 1]?.role === "assistant") {
        chat_history.push([messages[i].content, messages[i + 1].content])
      }
    }

    let fullAnswer = ""
    try {
      const response = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage.content,
          collection_name: collection,
          chat_history,
        }),
      })

      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      outer: while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const data = line.slice(6)
          if (data === "[DONE]") break outer
          try {
            const parsed = JSON.parse(data)
            if (parsed.type === "token") {
              fullAnswer += parsed.content
              setAnswer(fullAnswer)
            } else if (parsed.type === "error") {
              setError(parsed.content)
            }
          } catch {
            // Ignore malformed events and continue streaming
          }
        }
      }
    } catch (err) {
      setError(err.message || "Query failed")
    } finally {
      if (fullAnswer) {
        setMessages(prev => [...prev, { role: "assistant", content: fullAnswer }])
      }
      setAnswer("")
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>RAG Q&A</h1>

      <div className="upload-section">
        <label className="upload-btn">
          Upload PDF
          <input type="file" accept=".pdf" onChange={uploadPDF} hidden />
        </label>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="chat-window">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <p>{msg.content}</p>
          </div>
        ))}
        {answer && (
          <div className="message assistant">
            <p>{answer}</p>
          </div>
        )}
      </div>

      <div className="input-row">
        <input
          type="text"
          placeholder="Ask a question..."
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && askQuestion()}
        />
        <button onClick={askQuestion} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  )
}

export default App
