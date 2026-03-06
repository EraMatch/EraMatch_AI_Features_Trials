import React from 'react'
import { Trophy, Zap, ShieldCheck, BookOpen, Clock, AlertCircle } from 'lucide-react'

const BenchmarkLeaderboard = ({ results, onBack }) => {
  if (!results || results.length === 0) return null

  const successful = results.filter(r => r.status === 'success')
  const failed = results.filter(r => r.status !== 'success')

  const getClass = (score) => {
    if (score >= 8) return 'excellent'
    if (score >= 6) return 'good'
    return 'needs-improvement'
  }

  return (
    <div className="lb-container glass">
      <div className="lb-header">
        <div className="lb-title-row">
          <Trophy className="icon-gold" size={32} />
          <h2>AI Model Leaderboard</h2>
        </div>
        <p className="lb-subtitle">Benchmarked across Faithfulness, Quality, and Pedagogical Value</p>
        <span className="lb-eval-note">⚡ One fixed evaluator model judges all outputs — no self-scoring bias</span>
        <button className="btn btn-outline btn-sm lb-back" onClick={onBack}>← Back</button>
      </div>

      <div className="lb-table-wrap">
        <table className="lb-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Overall Score</th>
              <th>RAGAS Faithfulness</th>
              <th>DeepEval Quality</th>
              <th>QUEST Pedagogy</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {successful.map((r, i) => (
              <tr key={r.model} className={i === 0 ? 'top-rank' : ''}>
                <td>
                  <span className="rank-num">#{i + 1}</span>
                  <span className="model-name">{r.model} {i === 0 ? '👑' : ''}</span>
                </td>
                <td><span className={`score-badge ${getClass(r.overall_quality_score)}`}>{r.overall_quality_score}/10</span></td>
                <td><span className="metric"><ShieldCheck size={14} /> {Math.round(r.avg_faithfulness * 100)}%</span></td>
                <td><span className="metric"><Zap size={14} /> {r.avg_distractor_quality}/10</span></td>
                <td><span className="metric"><BookOpen size={14} /> {Math.round(r.avg_pedagogy_quest * 20)}%</span></td>
                <td><span className="metric muted"><Clock size={14} /> {r.latency_sec}s</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {failed.length > 0 && (
        <div className="failed-section">
          <h4>Failed Models ({failed.length})</h4>
          {failed.map(f => (
            <div key={f.model} className="failed-item">
              <AlertCircle size={14} />
              <span className="failed-model">{f.model}</span>
              <span className="failed-reason">{f.reason || 'Unknown error'} ({f.latency_sec}s)</span>
            </div>
          ))}
        </div>
      )}

      {successful.length > 0 && (
        <div className="rec-box">
          <AlertCircle size={20} />
          <span>
            <strong>Best Choice: </strong>
            <span className="rec-highlight">{successful[0].model}</span>
            {' '}— Score: {successful[0].overall_quality_score}/10 | Latency: {successful[0].latency_sec}s
          </span>
        </div>
      )}

      <style jsx>{`
        .lb-container { padding: 2.5rem; margin-top: 2rem; animation: slideUp 0.5s ease; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }

        .lb-header { position: relative; margin-bottom: 2rem; display: flex; flex-direction: column; gap: 0.5rem; }
        .lb-title-row { display: flex; align-items: center; gap: 1rem; }
        .icon-gold { color: #FFD700; filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.4)); }
        .lb-subtitle { color: var(--text-muted); font-size: 0.95rem; }
        .lb-eval-note { font-size: 0.75rem; color: var(--primary); background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3); padding: 0.3rem 0.75rem; border-radius: 4px; width: fit-content; }
        .lb-back { position: absolute; top: 0; right: 0; }

        .lb-table-wrap { overflow-x: auto; background: rgba(0,0,0,0.2); border-radius: 1rem; border: 1px solid var(--border); }
        .lb-table { width: 100%; border-collapse: collapse; text-align: left; }
        .lb-table th { padding: 1.25rem 1.5rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); border-bottom: 1px solid var(--border); }
        .lb-table td { padding: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); display: table-cell; }
        .top-rank { background: rgba(99,102,241,0.05); }
        .rank-num { font-weight: 900; color: var(--accent); font-size: 1.1rem; margin-right: 0.75rem; }
        .model-name { font-weight: 600; }

        .score-badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 2rem; font-weight: 800; font-size: 0.85rem; }
        .excellent { background: rgba(34,197,94,0.2); color: var(--success); border: 1px solid var(--success); }
        .good { background: rgba(99,102,241,0.2); color: var(--primary); border: 1px solid var(--primary); }
        .needs-improvement { background: rgba(239,68,68,0.1); color: var(--error); border: 1px solid var(--error); }

        .metric { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.9rem; }
        .metric.muted { color: var(--text-muted); }

        .failed-section { margin-top: 2rem; }
        .failed-section h4 { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.75rem; }
        .failed-item { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.2); border-radius: 0.5rem; font-size: 0.85rem; color: var(--error); margin-bottom: 0.5rem; }
        .failed-model { font-weight: 700; }
        .failed-reason { color: var(--text-muted); margin-left: auto; }

        .rec-box { margin-top: 2rem; padding: 1.5rem; background: rgba(99,102,241,0.1); border: 1px solid var(--primary); border-radius: 1rem; display: flex; gap: 1.5rem; align-items: center; font-size: 0.95rem; color: var(--text-muted); }
        .rec-highlight { color: var(--accent); font-weight: 800; }
      `}</style>
    </div>
  )
}

export default BenchmarkLeaderboard
