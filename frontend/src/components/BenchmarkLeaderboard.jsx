import React from 'react'
import { Trophy, Zap, ShieldCheck, BookOpen, Clock, AlertCircle } from 'lucide-react'

const BenchmarkLeaderboard = ({ results, onBack }) => {
    if (!results || results.length === 0) return null;

    const getPerformanceClass = (score) => {
        if (score >= 8) return 'excellent';
        if (score >= 6) return 'good';
        return 'needs-improvement';
    };

    return (
        <div className="leaderboard-container glass animation-slide-up">
            <div className="leaderboard-header">
                <div className="title-row">
                    <Trophy className="icon-gold" size={32} />
                    <h2>AI Model Leaderboard</h2>
                </div>
                <p className="subtitle">Benchmarked across Faithfulness, Quality, and Pedagogical Value</p>
                <button className="btn btn-outline btn-sm" onClick={onBack}>Back to Dashboard</button>
            </div>

            <div className="leaderboard-table-wrapper">
                <table className="leaderboard-table">
                    <thead>
                        <tr>
                            <th>Model Rank</th>
                            <th>Overall Score</th>
                            <th>RAGAS (Grounding)</th>
                            <th>DeepEval (Quality)</th>
                            <th>QUEST (Pedagogy)</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results.map((res, idx) => (
                            <tr key={res.model} className={idx === 0 ? 'top-rank' : ''}>
                                <td>
                                    <div className="rank-cell">
                                        <span className="rank-number">#{idx + 1}</span>
                                        <span className="model-name">{res.model} {idx === 0 && '👑'}</span>
                                    </div>
                                </td>
                                <td>
                                    <div className={`score-badge ${getPerformanceClass(res.overall_quality_score)}`}>
                                        {res.overall_quality_score}/10
                                    </div>
                                </td>
                                <td>
                                    <div className="metric-cell">
                                        <ShieldCheck size={14} />
                                        {Math.round(res.avg_faithfulness * 100)}%
                                    </div>
                                </td>
                                <td>
                                    <div className="metric-cell">
                                        <Zap size={14} />
                                        {res.avg_distractor_quality}/10
                                    </div>
                                </td>
                                <td>
                                    <div className="metric-cell">
                                        <BookOpen size={14} />
                                        {Math.round(res.avg_pedagogy_quest * 20)}%
                                    </div>
                                </td>
                                <td>
                                    <div className="metric-cell latency">
                                        <Clock size={14} />
                                        {res.latency_sec}s
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="recommendation-box">
                <AlertCircle size={20} />
                <div>
                    <strong>Best Choice:</strong> <span className="highlight-text">{results[0].model}</span> is currently your top performing model.
                    It offers the best balance of {results[0].avg_faithfulness > 0.8 ? 'high factual accuracy' : 'speed'} and pedagogical structure.
                </div>
            </div>

            <style jsx>{`
        .leaderboard-container {
          padding: 2.5rem;
          margin-top: 2rem;
        }
        .leaderboard-header {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          margin-bottom: 2.5rem;
          position: relative;
        }
        .leaderboard-header .btn {
          position: absolute;
          right: 0;
          top: 0;
        }
        .title-row {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        .icon-gold { color: #FFD700; filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.4)); }
        .subtitle { color: var(--text-muted); font-size: 0.95rem; }

        .leaderboard-table-wrapper {
          overflow-x: auto;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 1rem;
          border: 1px solid var(--border);
        }
        .leaderboard-table {
          width: 100%;
          border-collapse: collapse;
          text-align: left;
        }
        .leaderboard-table th {
          padding: 1.25rem 1.5rem;
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: var(--text-muted);
          border-bottom: 1px solid var(--border);
        }
        .leaderboard-table td {
          padding: 1.5rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .top-rank {
          background: rgba(99, 102, 241, 0.05);
        }
        .rank-cell {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        .rank-number {
          font-weight: 900;
          color: var(--accent);
          font-size: 1.1rem;
        }
        .model-name {
          font-weight: 600;
          color: var(--text);
        }

        .score-badge {
          display: inline-block;
          padding: 0.4rem 1rem;
          border-radius: 2rem;
          font-weight: 800;
          font-size: 0.85rem;
        }
        .excellent { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }
        .good { background: rgba(99, 102, 241, 0.2); color: var(--primary); border: 1px solid var(--primary); }
        .needs-improvement { background: rgba(239, 68, 68, 0.1); color: var(--error); border: 1px solid var(--error); }

        .metric-cell {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
          color: var(--text);
        }
        .metric-cell.latency { color: var(--text-muted); }

        .recommendation-box {
          margin-top: 2rem;
          padding: 1.5rem;
          background: rgba(99, 102, 241, 0.1);
          border: 1px solid var(--primary);
          border-radius: 1rem;
          display: flex;
          gap: 1.5rem;
          align-items: center;
          font-size: 0.95rem;
          color: var(--text-muted);
        }
        .highlight-text {
          color: var(--accent);
          font-weight: 800;
        }
        
        .animation-slide-up {
          animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
        </div>
    )
}

export default BenchmarkLeaderboard
