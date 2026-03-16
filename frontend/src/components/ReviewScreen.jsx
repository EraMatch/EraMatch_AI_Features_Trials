import React, { useState } from 'react'
import { Check, Trash2, AlertCircle, Edit2, Save } from 'lucide-react'

const ReviewScreen = ({ questions: initialQuestions, onSave, onCancel }) => {
  const [questions, setQuestions] = useState(initialQuestions)
  const [editingId, setEditingId] = useState(null)
  const [expandedEval, setExpandedEval] = useState(null)

  const handleDelete = (index) => {
    setQuestions(questions.filter((_, i) => i !== index))
  }

  const handleUpdate = (index, field, value) => {
    setQuestions(questions.map((q, i) => {
      if (i === index) {
        if (field === 'options') {
          return { ...q, question_config: { ...q.question_config, options: value } }
        }
        if (field === 'evidence') {
          return { ...q, question_config: { ...q.question_config, evidence: value } }
        }
        return { ...q, [field]: value }
      }
      return q
    }))
  }

  const toggleMcqCorrect = (idx, oIdx) => {
    const q = questions[idx];
    let current = Array.isArray(q.correct_answer) ? [...q.correct_answer] : [];
    if (current.includes(oIdx)) {
      current = current.filter(i => i !== oIdx);
    } else {
      current.push(oIdx);
    }
    handleUpdate(idx, 'correct_answer', current);
  }

  return (
    <div className="review-container">
      <div className="review-header">
        <h2>Review Generated Questions</h2>
        <div className="actions">
          <button className="btn btn-outline" onClick={onCancel}>Cancel</button>
          <button className="btn btn-primary" onClick={() => onSave(questions)}>
            <Check size={20} />
            Approve & Import {questions.length} Questions
          </button>
        </div>
      </div>

      <div className="questions-list">
        {questions.some(q => q.eval_results?.ragas_faithfulness < 0.7 || (q.question_type === 'mcq' && q.eval_results?.geval_score < 6)) && (
          <div className="global-warning-banner glass animate-pulse">
            <AlertCircle size={18} />
            <span>Found {questions.filter(q => q.eval_results?.ragas_faithfulness < 0.7 || (q.question_type === 'mcq' && q.eval_results?.geval_score < 6)).length} questions with low quality scores. Verification recommended.</span>
          </div>
        )}
        {questions.map((q, idx) => {
          const isMcq = q.question_type === 'mcq'
          const options = q.question_config?.options || []
          const evidence = q.question_config?.evidence || ''
          const isMissingAnswer = isMcq
            ? (!q.correct_answer || (Array.isArray(q.correct_answer) && q.correct_answer.length === 0))
            : !q.correct_answer;

          const evalRes = q.eval_results || {};
          const isLowQuality = evalRes.ragas_faithfulness < 0.7 || (isMcq && evalRes.geval_score < 6);

          return (
            <div key={idx} className={`question-card glass ${isMissingAnswer ? 'invalid' : ''} ${isLowQuality ? 'flagged' : ''}`}>
              <div className="card-header">
                <span className={`badge ${q.question_type}`}>{q.question_type.toUpperCase()}</span>
                <span className="difficulty">Level: {q.difficulty}</span>
                {q.points && <span className="points">{q.points}pts</span>}
                {isLowQuality && <span className="quality-warning pulse-slow"><AlertCircle size={14} /> Low Quality Flag</span>}
                <div className="card-actions">
                  <button className="icon-btn" onClick={() => setEditingId(idx === editingId ? null : idx)}>
                    <Edit2 size={16} />
                  </button>
                  <button className="icon-btn delete" onClick={() => handleDelete(idx)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>

              {editingId === idx ? (
                <div className="edit-form">
                  <label>Question Text</label>
                  <textarea
                    value={q.question_text}
                    onChange={(e) => handleUpdate(idx, 'question_text', e.target.value)}
                    className="edit-input"
                  />

                  {isMcq ? (
                    <div className="options-edit">
                      <label>Options (Select all correct answers)</label>
                      {options.map((opt, oIdx) => (
                        <div key={oIdx} className="option-row">
                          <input
                            type="checkbox"
                            checked={Array.isArray(q.correct_answer) && q.correct_answer.includes(oIdx)}
                            onChange={() => toggleMcqCorrect(idx, oIdx)}
                          />
                          <input
                            type="text"
                            value={opt}
                            onChange={(e) => {
                              const newOpts = [...options]
                              newOpts[oIdx] = e.target.value
                              handleUpdate(idx, 'options', newOpts)
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="answer-edit">
                      <label>Reference Answer</label>
                      <textarea
                        value={q.correct_answer || ''}
                        onChange={(e) => handleUpdate(idx, 'correct_answer', e.target.value)}
                        className="edit-input"
                        placeholder="Model answer for this essay..."
                      />
                    </div>
                  )}

                  <div className="evidence-edit">
                    <label>Source Evidence (Grounding)</label>
                    <textarea
                      value={evidence}
                      onChange={(e) => handleUpdate(idx, 'evidence', e.target.value)}
                      className="edit-input"
                      placeholder="Citation from source..."
                    />
                  </div>

                  <button className="btn btn-primary btn-sm" onClick={() => setEditingId(null)}>Done</button>
                </div>
              ) : (
                <div className="display-content">
                  <h3 className="title">{q.question_text}</h3>
                  {isMcq && (
                    <ul className="options-list">
                      {options.map((opt, oIdx) => {
                        const isCorrect = Array.isArray(q.correct_answer)
                          ? q.correct_answer.includes(oIdx)
                          : q.correct_answer === oIdx;
                        return (
                          <li key={oIdx} className={isCorrect ? 'correct' : ''}>
                            {opt}
                            {isCorrect && <Check size={14} className="check-icon" />}
                          </li>
                        );
                      })}
                    </ul>
                  )}

                  {!isMcq && q.correct_answer && (
                    <div className="reference-answer">
                      <strong>Reference Answer:</strong>
                      <p>{q.correct_answer}</p>
                    </div>
                  )}

                  {evidence && (
                    <div className="evidence-box">
                      <strong>Source Evidence:</strong>
                      <p>"{evidence}"</p>
                    </div>
                  )}

                  {isLowQuality && (
                    <div className="auto-warning-msg">
                      <strong>AI Warning:</strong> Evaluation scores indicate this question may be hallucinated or poorly structured. Please review carefully.
                    </div>
                  )}

                  {q.eval_results && (
                    <div className="evaluation-section">
                      <div className="eval-summary-header" onClick={() => setExpandedEval(expandedEval === idx ? null : idx)}>
                        <div className="eval-pill-row">
                          <div className={`eval-pill small ${q.eval_results.ragas_faithfulness > 0.8 ? 'good' : 'bad'}`}>
                            <span className="label">Faithfulness:</span>
                            <span className="score">{Math.round(q.eval_results.ragas_faithfulness * 100)}%</span>
                          </div>
                          {isMcq && (
                            <div className={`eval-pill small ${q.eval_results.geval_score > 7 ? 'good' : 'bad'}`}>
                              <span className="label">Quality:</span>
                              <span className="score">{q.eval_results.geval_score}/10</span>
                            </div>
                          )}
                        </div>
                        <button className="expand-eval-btn">
                          {expandedEval === idx ? 'Hide Report' : 'View Detailed Evaluation'}
                        </button>
                      </div>

                      {expandedEval === idx && (
                        <div className="evaluation-report animation-slide-down">
                          <div className="report-pillar">
                            <h4>Pillar 1: Faithfulness (RAGAS)</h4>
                            <p className="reasoning">{q.eval_results.ragas_reasoning}</p>
                          </div>

                          {isMcq && (
                            <div className="report-pillar">
                              <h4>Pillar 2: MCQ Distractors (DeepEval)</h4>
                              <p className="reasoning">{q.eval_results.geval_reasoning}</p>
                            </div>
                          )}

                          <div className="report-pillar">
                            <h4>Pillar 3: Pedagogical Analysis (QUEST)</h4>
                            <div className="quest-details">
                              {q.eval_results.quest && !q.eval_results.quest.error && (
                                Object.entries(q.eval_results.quest).filter(([k]) => k !== 'overall_pedagogy_score' && k !== 'error').map(([key, data]) => (
                                  <div key={key} className="quest-detail-item">
                                    <div className="dim-header">
                                      <span className="dim-name">{key.toUpperCase()}</span>
                                      <div className="dim-dots">
                                        {[1, 2, 3, 4, 5].map(dot => (
                                          <span key={dot} className={`dot ${data.score >= dot ? 'active' : ''}`}></span>
                                        ))}
                                      </div>
                                    </div>
                                    <p className="dim-feedback">{data.feedback}</p>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="tags-row">
                    {q.tags?.map(tag => <span key={tag} className="tag">#{tag}</span>)}
                  </div>
                </div>
              )}

              {isMissingAnswer && (
                <div className="error-msg">
                  <AlertCircle size={14} />
                  <span>Missing correct answer! Please select one.</span>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <style jsx>{`
        .review-container {
          animation: slideUp 0.3s ease-out;
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .review-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
        }
        .actions {
          display: flex;
          gap: 1rem;
        }
        .questions-list {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }
        .question-card.flagged {
          border-color: var(--warning);
          box-shadow: 0 0 15px rgba(245, 158, 11, 0.1);
        }
        .quality-warning {
          display: flex;
          align-items: center;
          gap: 0.4rem;
          font-size: 0.7rem;
          font-weight: 700;
          color: var(--warning);
          background: rgba(245, 158, 11, 0.1);
          padding: 0.2rem 0.6rem;
          border-radius: 4rem;
          text-transform: uppercase;
        }
        .auto-warning-msg {
          margin-top: 1rem;
          padding: 1rem;
          background: rgba(239, 68, 68, 0.05);
          border: 1px solid rgba(239, 68, 68, 0.2);
          border-radius: 0.5rem;
          font-size: 0.85rem;
          color: var(--error);
        }
        .global-warning-banner {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1rem 1.5rem;
          background: rgba(245, 158, 11, 0.05);
          border: 1px solid var(--warning);
          border-radius: 0.75rem;
          color: var(--warning);
          font-size: 0.9rem;
          font-weight: 500;
        }
        .pulse-slow {
          animation: pulse 2s infinite;
        }
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.6; }
          100% { opacity: 1; }
        }
        .card-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 1rem;
        }
        .badge {
          padding: 0.2rem 0.6rem;
          border-radius: 4px;
          font-size: 0.7rem;
          font-weight: 700;
        }
        .badge.mcq { background: rgba(99, 102, 241, 0.2); color: var(--primary); }
        .badge.essay { background: rgba(192, 132, 252, 0.2); color: var(--accent); }
        .difficulty {
          font-size: 0.8rem;
          color: var(--text-muted);
        }
        .card-actions {
          margin-left: auto;
          display: flex;
          gap: 0.5rem;
        }
        .icon-btn {
          background: transparent;
          border: none;
          color: var(--text-muted);
          padding: 4px;
          border-radius: 4px;
          transition: background 0.2s;
        }
        .icon-btn:hover { background: rgba(255, 255, 255, 0.05); }
        .icon-btn.delete:hover { color: var(--error); background: rgba(239, 68, 68, 0.1); }
        
        .title { margin-bottom: 1rem; font-size: 1.1rem; }
        .options-list { list-style: none; display: flex; flex-direction: column; gap: 0.5rem; }
        .options-list li {
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid var(--border);
          border-radius: 0.5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.9rem;
        }
        .options-list li.correct {
          border-color: var(--success);
          background: rgba(34, 197, 94, 0.1);
        }
        .check-icon { color: var(--success); }

        .tags-row {
          margin-top: 1rem;
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
        }
        .tag {
          font-size: 0.75rem;
          color: var(--accent);
          background: rgba(192, 132, 252, 0.1);
          padding: 0.2rem 0.5rem;
          border-radius: 4px;
        }
        .points {
          font-size: 0.8rem;
          background: rgba(255, 255, 255, 0.05);
          padding: 0.2rem 0.6rem;
          border-radius: 4px;
          color: var(--text-muted);
        }
        .error-msg {
          margin-top: 1rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: var(--error);
          font-size: 0.85rem;
          font-weight: 500;
        }

        .evaluation-section {
          margin-top: 1.5rem;
          padding-top: 1rem;
          border-top: 1px solid var(--border);
        }
        .eval-summary-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          cursor: pointer;
          padding: 0.5rem;
          border-radius: 0.5rem;
          transition: background 0.2s;
        }
        .eval-summary-header:hover {
          background: rgba(255,255,255,0.03);
        }
        .expand-eval-btn {
          background: transparent;
          border: 1px solid var(--border);
          color: var(--accent);
          font-size: 0.75rem;
          padding: 0.3rem 0.8rem;
          border-radius: 4px;
          cursor: pointer;
        }
        .eval-pill-row {
          display: flex;
          gap: 0.75rem;
        }
        .eval-pill.small {
          padding: 0.25rem 0.75rem;
          border-radius: 1rem;
          border: 1px solid var(--border);
          font-size: 0.75rem;
          background: rgba(255,255,255,0.02);
        }
        .eval-pill .label { color: var(--text-muted); margin-right: 0.4rem; }
        .eval-pill .score { font-weight: 700; }
        .eval-pill.good .score { color: var(--success); }
        .eval-pill.bad .score { color: var(--error); }

        .evaluation-report {
          margin-top: 1rem;
          background: rgba(0,0,0,0.2);
          border-radius: 0.75rem;
          padding: 1.25rem;
          border: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
        }
        .animation-slide-down {
          animation: slideDown 0.3s ease-out;
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .report-pillar h4 {
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--text-muted);
          margin-bottom: 0.6rem;
        }
        .reasoning {
          font-size: 0.9rem;
          line-height: 1.5;
          color: var(--text);
          background: rgba(255,255,255,0.03);
          padding: 0.75rem;
          border-radius: 0.5rem;
        }
        .quest-details {
          display: grid;
          grid-template-columns: 1fr;
          gap: 1rem;
        }
        .quest-detail-item {
          background: rgba(255,255,255,0.02);
          padding: 0.75rem;
          border-radius: 0.5rem;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .dim-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }
        .dim-name {
          font-size: 0.7rem;
          font-weight: 800;
          color: var(--accent);
        }
        .dim-dots { display: flex; gap: 3px; }
        .dim-feedback {
          font-size: 0.85rem;
          color: var(--text-muted);
          line-height: 1.4;
        }
        .dot {
          width: 10px;
          height: 3px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
        }
        .dot.active {
          background: var(--accent);
          box-shadow: 0 0 8px var(--accent);
        }

        .reference-answer {
          margin-top: 1rem;
          padding: 1rem;
          background: rgba(192, 132, 252, 0.05);
          border-left: 3px solid var(--accent);
          border-radius: 0 0.5rem 0.5rem 0;
          font-size: 0.9rem;
        }
        .reference-answer strong {
          display: block;
          margin-bottom: 0.5rem;
          color: var(--accent);
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .evidence-box {
          margin-top: 1rem;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px dashed var(--border);
          border-radius: 0.5rem;
          font-size: 0.85rem;
          font-style: italic;
          color: var(--text-muted);
        }
        .evidence-box strong {
          display: block;
          margin-bottom: 0.5rem;
          color: var(--text-muted);
          font-size: 0.75rem;
          font-style: normal;
          text-transform: uppercase;
        }

        .edit-form { display: flex; flex-direction: column; gap: 1rem; }
        .edit-form label {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--text-muted);
          margin-bottom: -0.5rem;
        }
        .edit-input {
          width: 100%;
          background: var(--bg);
          border: 1px solid var(--border);
          color: var(--text);
          padding: 0.75rem;
          border-radius: 0.5rem;
          min-height: 80px;
        }
        .options-edit { display: flex; flex-direction: column; gap: 0.5rem; }
        .option-row { display: flex; align-items: center; gap: 0.75rem; }
        .option-row input[type="text"] {
          flex: 1;
          background: var(--bg);
          border: 1px solid var(--border);
          color: var(--text);
          padding: 0.4rem 0.75rem;
          border-radius: 0.4rem;
        }
        .btn-sm { padding: 0.4rem 1rem; font-size: 0.85rem; width: fit-content; }
      `}</style>
    </div>
  )
}

export default ReviewScreen
