import { useState } from 'react'
import './App.css'
import axios from 'axios'

const API = "http://localhost:8000"

function App() {
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState([])
  const [answer, setAnswer] = useState("")
  const [loading, setLoading] = useState(false)
  const [collection, setCollection] = useState("default")

  const uploadPDF = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    const formData = new FormData()
    formData.append("file", file)
    await axios.post(`${API}/ingest/pdf?collection_name=${collection}`, formData)
    alert(`${file.name} uploaded successfully`)
  }

  const askQuestion = async () => {
    if (!question.trim() || loading) return
    setLoading(true)
    const userMessage = { role: "user", content: question }
    setMessages(prev => [...prev, userMessage])
    setQuestion("")
    setAnswer("")

    const response = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: userMessage.content,
        collection_name: collection,
        chat_history: messages
          .filter((_, i) => i % 2 === 0)
          .map((m, i) => [m.content, messages[i * 2 + 1]?.content ?? ""])
      })
    })

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let fullAnswer = ""

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value)
      const lines = text.split("\n").filter(l => l.startsWith("data: "))
      for (const line of lines) {
        const data = line.slice(6)
        if (data === "[DONE]") break
        const parsed = JSON.parse(data)
        if (parsed.type === "token") {
          fullAnswer += parsed.content
          setAnswer(fullAnswer)
        }
      }
    }

    setMessages(prev => [...prev, { role: "assistant", content: fullAnswer }])
    setAnswer("")
    setLoading(false)
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
