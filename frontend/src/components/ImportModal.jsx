import React, { useState } from 'react'
import { X, Upload, Sparkles, FileText, ChevronRight, ArrowLeft } from 'lucide-react'

const ImportModal = ({ onClose, onSelect, isProcessing }) => {
  const [step, setStep] = useState('selection'); // 'selection' or 'config'
  const [selectedType, setSelectedType] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [mcqCount, setMcqCount] = useState(10);
  const [essayCount, setEssayCount] = useState(0);
  const [difficulty, setDifficulty] = useState('Medium');

  const handleFileChange = (e, type) => {
    const file = e.target.files[0];
    if (file) {
      if (type === 'generate') {
        setSelectedType(type);
        setSelectedFile(file);
        setStep('config');
      } else {
        onSelect(type, file);
      }
    }
  };

  const handleConfirm = () => {
    onSelect(selectedType, selectedFile, { mcqCount, essayCount, difficulty });
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content glass">
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {step === 'config' && !isProcessing && (
              <button className="back-btn" onClick={() => setStep('selection')}><ArrowLeft size={20} /></button>
            )}
            <h2>{step === 'selection' ? 'Import Questions' : 'Configure Generation'}</h2>
          </div>
          <button className="close-btn" onClick={onClose} disabled={isProcessing}><X size={20} /></button>
        </div>

        {isProcessing ? (
          <div className="processing-state">
            <div className="spinner"></div>
            <p>AI is processing your material...</p>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>This usually takes 10-30 seconds</span>
          </div>
        ) : step === 'selection' ? (
          <div className="import-options">
            <label className="option-card">
              <input
                type="file"
                accept=".csv,.xlsx"
                style={{ display: 'none' }}
                onChange={(e) => handleFileChange(e, 'sheet')}
              />
              <div className="option-icon"><Upload size={24} /></div>
              <div className="option-info">
                <h3>Upload Question Sheet</h3>
                <p>Deterministic import from .csv or .xlsx files.</p>
              </div>
              <ChevronRight size={20} className="chevron" />
            </label>

            <label className="option-card">
              <input
                type="file"
                accept=".pdf,.docx,.txt"
                style={{ display: 'none' }}
                onChange={(e) => handleFileChange(e, 'generate')}
              />
              <div className="option-icon" style={{ background: 'var(--accent)' }}><Sparkles size={24} /></div>
              <div className="option-info">
                <h3>AI Generative Import</h3>
                <p>Generate new questions from PDF, Docx, or Text.</p>
              </div>
              <ChevronRight size={20} className="chevron" />
            </label>

            <label className="option-card">
              <input
                type="file"
                accept=".pdf,.md,.txt"
                style={{ display: 'none' }}
                onChange={(e) => handleFileChange(e, 'extract')}
              />
              <div className="option-icon" style={{ background: 'var(--success)' }}><FileText size={24} /></div>
              <div className="option-info">
                <h3>AI Extraction Import</h3>
                <p>Extract existing questions from an old exam paper or MD.</p>
              </div>
              <ChevronRight size={20} className="chevron" />
            </label>
          </div>
        ) : (
          <div className="config-form">
            <div className="form-group">
              <label>File Selected</label>
              <div className="file-preview">
                <FileText size={16} />
                <span>{selectedFile?.name}</span>
              </div>
            </div>

            <div className="form-group">
              <label>Question Distribution (Total: {mcqCount + essayCount})</label>
              <div className="distribution-controls">
                <div className="control-item">
                  <span>MCQs</span>
                  <div className="counter">
                    <button onClick={() => setMcqCount(Math.max(0, mcqCount - 1))}>-</button>
                    <input type="number" value={mcqCount} readOnly />
                    <button onClick={() => setMcqCount(Math.min(20, mcqCount + 1))}>+</button>
                  </div>
                </div>
                <div className="control-item">
                  <span>Essays</span>
                  <div className="counter">
                    <button onClick={() => setEssayCount(Math.max(0, essayCount - 1))}>-</button>
                    <input type="number" value={essayCount} readOnly />
                    <button onClick={() => setEssayCount(Math.min(10, essayCount + 1))}>+</button>
                  </div>
                </div>
              </div>
              {mcqCount + essayCount > 25 && (
                <p style={{ color: 'var(--error)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                  Warning: High question counts may lead to AI timeouts.
                </p>
              )}
            </div>

            <div className="form-group">
              <label>Difficulty Level</label>
              <div className="difficulty-selector">
                {['Easy', 'Medium', 'Hard'].map(d => (
                  <button
                    key={d}
                    className={difficulty === d ? 'active' : ''}
                    onClick={() => setDifficulty(d)}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </div>

            <button className="btn btn-primary generate-btn" onClick={handleConfirm}>
              <Sparkles size={18} />
              Generate Questions
            </button>
          </div>
        )}
      </div>

      <style jsx>{`
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          backdrop-filter: blur(4px);
        }
        .modal-content {
          width: 500px;
          padding: 2.5rem;
          position: relative;
        }
        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
        }
        .close-btn, .back-btn {
          background: transparent;
          border: none;
          color: var(--text-muted);
          cursor: pointer;
          padding: 0.5rem;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
        }
        .close-btn:hover, .back-btn:hover {
          background: rgba(255, 255, 255, 0.1);
          color: var(--text);
        }
        .import-options {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        .option-card {
          display: flex;
          align-items: center;
          padding: 1.25rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid var(--border);
          border-radius: 0.75rem;
          cursor: pointer;
          transition: all 0.2s;
        }
        .option-card:hover {
          background: rgba(255, 255, 255, 0.07);
          border-color: var(--primary);
          transform: translateX(4px);
        }
        .option-icon {
          width: 48px;
          height: 48px;
          background: var(--primary);
          border-radius: 0.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-right: 1.25rem;
        }
        .option-info {
          flex: 1;
        }
        .option-info h3 {
          font-size: 1.1rem;
          margin-bottom: 0.25rem;
        }
        .option-info p {
          font-size: 0.9rem;
          color: var(--text-muted);
        }
        .chevron {
          color: var(--text-muted);
          opacity: 0.5;
        }

        /* Config Form Styles */
        .config-form {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }
        .form-group label {
          display: block;
          margin-bottom: 0.75rem;
          font-size: 0.9rem;
          font-weight: 500;
          color: var(--text-muted);
        }
        .file-preview {
          background: rgba(255, 255, 255, 0.05);
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          display: flex;
          align-items: center;
          gap: 0.75rem;
          font-size: 0.9rem;
          border: 1px dashed var(--border);
        }
        .distribution-controls {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        .control-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: rgba(255, 255, 255, 0.03);
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
        }
        .counter {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .counter button {
          width: 32px;
          height: 32px;
          border-radius: 4px;
          border: 1px solid var(--border);
          background: rgba(255, 255, 255, 0.1);
          color: var(--text);
          cursor: pointer;
        }
        .counter input {
          width: 40px;
          text-align: center;
          background: transparent;
          border: none;
          color: var(--text);
          font-weight: 700;
        }
        .difficulty-selector {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 0.5rem;
        }
        .difficulty-selector button {
          padding: 0.75rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid var(--border);
          border-radius: 0.5rem;
          color: var(--text);
          cursor: pointer;
          transition: all 0.2s;
          font-weight: 600;
        }
        .difficulty-selector button.active {
          background: var(--primary);
          border-color: var(--primary);
          box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        }
        .generate-btn {
          margin-top: 1rem;
          display: flex;
          justify-content: center;
          gap: 0.5rem;
          padding: 1rem;
        }

        .processing-state {
          text-align: center;
          padding: 3rem 0;
        }
        .spinner {
          width: 40px;
          height: 40px;
          border: 3px solid rgba(99, 102, 241, 0.2);
          border-top-color: var(--primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 1.5rem;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

export default ImportModal
