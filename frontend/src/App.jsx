import React, { useState, useEffect } from 'react'
import { Plus, Upload, Sparkles, FileText, ChevronRight, X, Loader2, Clock, Zap } from 'lucide-react'
import ImportModal from './components/ImportModal'
import ReviewScreen from './components/ReviewScreen'
import BenchmarkLeaderboard from './components/BenchmarkLeaderboard'

function App() {
    const [showImportModal, setShowImportModal] = useState(false)
    const [stagedQuestions, setStagedQuestions] = useState([])
    const [isProcessing, setIsProcessing] = useState(false)
    const [jobId, setJobId] = useState(null)
    const [jobStatus, setJobStatus] = useState(null)
    const [error, setError] = useState(null)
    const [benchmarkResults, setBenchmarkResults] = useState(null)

    // Polling for background jobs
    useEffect(() => {
        let pollInterval;
        if (jobId) {
            pollInterval = setInterval(async () => {
                try {
                    const res = await fetch(`/api/jobs/${jobId}`)
                    const data = await res.json()
                    setJobStatus(data.status)

                    if (data.status === 'completed') {
                        if (Array.isArray(data.result) && data.result.length > 0 && 'latency_sec' in data.result[0]) {
                            setBenchmarkResults(data.result)
                        } else {
                            setStagedQuestions(data.result)
                        }
                        setJobId(null)
                        setIsProcessing(false)
                        clearInterval(pollInterval)
                    } else if (data.status === 'failed') {
                        setError(data.error || 'Job failed')
                        setJobId(null)
                        setIsProcessing(false)
                        clearInterval(pollInterval)
                    }
                } catch (err) {
                    console.error('Polling error:', err)
                }
            }, 2000)
        }
        return () => clearInterval(pollInterval)
    }, [jobId])

    const handleBenchmark = async (force = false) => {
        setIsProcessing(true)
        setError(null)
        const formData = new FormData()
        formData.append('background', 'true')
        if (force) formData.append('force_regenerate', 'true')

        try {
            const response = await fetch('/api/import/benchmark', {
                method: 'POST',
                body: formData,
            })
            const data = await response.json()
            if (data.status === 'queued') {
                setJobId(data.job_id)
                setJobStatus('benchmarking')
            }
        } catch (err) {
            setError(err.message)
            setIsProcessing(false)
        }
    }

    const handleImportChoice = async (type, file, metadata = {}, background = false) => {
        setIsProcessing(true)
        setError(null)
        const formData = new FormData()
        formData.append('file', file)
        if (background) formData.append('background', 'true')

        let endpoint = `/api/import/${type}`
        if (type === 'generate') {
            formData.append('mcq_count', metadata.mcqCount || '10')
            formData.append('essay_count', metadata.essayCount || '0')
            formData.append('difficulty', metadata.difficulty || 'Medium')
        }

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Import failed')
            }

            const data = await response.json()

            if (data.status === 'large_file') {
                const proceed = window.confirm(`Large file detected (${data.page_count} pages). This may take a while. Proceed in background?`)
                if (proceed) {
                    return handleImportChoice(type, file, metadata, true)
                } else {
                    setIsProcessing(false)
                    return
                }
            }

            if (data.status === 'queued') {
                setJobId(data.job_id)
                setJobStatus('queued')
                setShowImportModal(false)
                return
            }

            setStagedQuestions(data.questions)
            setShowImportModal(false)
        } catch (err) {
            setError(err.message)
        } finally {
            if (!jobId) setIsProcessing(false)
        }
    }

    return (
        <div className="container">
            <header style={{ marginBottom: '3rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h1 style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent)' }}>Question Bank</h1>
                    <p style={{ color: 'var(--text-muted)' }}>Manage your assessment content</p>
                </div>
                <div>
                    {!isProcessing && stagedQuestions.length === 0 && !benchmarkResults && (
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            <button
                                className="btn btn-outline"
                                onClick={handleBenchmark}
                            >
                                <Zap size={20} />
                                Benchmark Models
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={() => setShowImportModal(true)}
                            >
                                <Plus size={20} />
                                Create New
                            </button>
                        </div>
                    )}
                </div>
            </header>

            {error && (
                <div className="glass" style={{ padding: '1rem', borderLeft: '4px solid var(--error)', marginBottom: '2rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <X color="var(--error)" onClick={() => setError(null)} style={{ cursor: 'pointer' }} />
                    <span>Error: {error}</span>
                </div>
            )}

            {isProcessing && jobId && (
                <div className="glass" style={{ padding: '3rem', textAlign: 'center' }}>
                    <Clock size={48} className="spin" style={{ marginBottom: '1rem', color: 'var(--primary)' }} />
                    <h2>Processing in Background</h2>
                    <p style={{ color: 'var(--text-muted)' }}>Status: <span style={{ color: 'var(--accent)', fontWeight: 700 }}>{jobStatus?.toUpperCase()}</span></p>
                    <p>You can wait here or come back later. We're extracting your questions...</p>
                </div>
            )}

            {benchmarkResults ? (
                <BenchmarkLeaderboard
                    results={benchmarkResults}
                    onBack={() => setBenchmarkResults(null)}
                    onRerun={() => { setBenchmarkResults(null); handleBenchmark(true) }}
                />
            ) : !isProcessing && stagedQuestions.length > 0 ? (
                <ReviewScreen
                    questions={stagedQuestions}
                    onSave={() => setStagedQuestions([])}
                    onCancel={() => setStagedQuestions([])}
                />
            ) : !isProcessing && !jobId && (
                <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-muted)' }} className="glass">
                    <FileText size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                    <p>Your question bank is empty. Click "Create New" to get started.</p>
                </div>
            )}

            {showImportModal && (
                <ImportModal
                    onClose={() => setShowImportModal(false)}
                    onSelect={handleImportChoice}
                    isProcessing={isProcessing && !jobId}
                />
            )}

            <style jsx>{`
        .spin { animation: rotate 2s linear infinite; }
        @keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
        </div>
    )
}

export default App
