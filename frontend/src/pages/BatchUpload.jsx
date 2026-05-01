import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Download, AlertCircle, CheckCircle, Eye, FileText, Sparkles, TrendingUp, Loader } from 'lucide-react';
import { toast } from 'react-hot-toast';
import Card from '../components/Card';
import Button from '../components/Button';
import Badge from '../components/Badge';
import LoadingSkeleton from '../components/LoadingSkeleton';
import ProgressBar from '../components/ProgressBar';

import AnimatedBackground from '../components/AnimatedBackground';
import { processBatch } from '../services/api';

export default function BatchUpload() {
  const [file, setFile] = useState(null);
  const [columnName, setColumnName] = useState('');
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = async (selectedFile) => {
    if (!selectedFile) return;

    setFile(selectedFile);
    setError(null);
    setLoading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return prev;
        }
        return prev + 10;
      });
    }, 100);

    try {
      const text = await selectedFile.text();
      const lines = text.split('\n').filter(line => line.trim());
      const header = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
      setColumns(header);

      const previewRows = lines.slice(1, 6).map(line =>
        line.split(',').map(cell => cell.trim().replace(/"/g, ''))
      );
      setPreview(previewRows);

      if (header.length > 0) {
        setColumnName(header[0]);
      }

      setUploadProgress(100);
      toast.success('File loaded successfully!');
    } catch (err) {
      setError('Could not parse file. Make sure it\'s a valid CSV.');
      setFile(null);
      toast.error('Failed to parse file');
    } finally {
      setTimeout(() => {
        setLoading(false);
        setUploadProgress(0);
      }, 500);
    }
  };

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files?.[0];
    await handleFile(selectedFile);
  };

  const handleProcess = async () => {
    if (!file || !columnName) {
      setError('Please select a file and column');
      toast.error('Please select a file and column');
      return;
    }

    setProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('column', columnName);

      const data = await processBatch(formData);
      setResults(data);
      toast.success(`Processed ${data.results.length} headlines!`);
    } catch (err) {
      setError(err.message || 'Failed to process batch');
      toast.error(err.message || 'Failed to process batch');
    } finally {
      setProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!results) return;

    const csv = results.results.map(row =>
      `"${row.headline}","${row.is_clickbait ? 'Clickbait' : 'Non-Clickbait'}","${(row.confidence * 100).toFixed(1)}%"`
    ).join('\n');

    const header = '"Headline","Clickbait","Confidence"\n';
    const blob = new Blob([header + csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'clickbait_results.csv';
    a.click();
    window.URL.revokeObjectURL(url);
    toast.success('Results downloaded!');
  };

  const getClickbaitStats = () => {
    if (!results) return { clickbait: 0, nonClickbait: 0, percentage: 0 };
    const clickbait = results.results.filter(r => r.is_clickbait).length;
    const total = results.results.length;
    return {
      clickbait,
      nonClickbait: total - clickbait,
      percentage: ((clickbait / total) * 100).toFixed(1)
    };
  };

  const stats = getClickbaitStats();

  return (
    <motion.main
      className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 min-h-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <AnimatedBackground />
      {/* Header */}
      <motion.div
        className="mb-8 md:mb-12"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <motion.div
            className="w-12 h-12 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl flex items-center justify-center"
            whileHover={{ rotate: 360, scale: 1.1 }}
            transition={{ duration: 0.6 }}
          >
            <FileText className="w-6 h-6 text-white" />
          </motion.div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-white">Batch Upload</h1>
            <p className="text-gray-200">
              Upload a CSV file with multiple headlines for batch analysis
            </p>
          </div>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6 md:gap-8">
        {/* Upload Section */}
        <div className="lg:col-span-2 space-y-6">
          {/* Drag & Drop Upload */}
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Card
              className={`p-8 md:p-12 border-2 border-dashed transition-all duration-300 bg-primary-900/30 backdrop-blur-sm relative overflow-hidden ${dragActive
                ? 'border-primary-400 bg-primary-800/50 shadow-glow-teal'
                : 'border-primary-600/40 hover:border-primary-500/60'
                }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {/* Animated Background */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-accent-500/10"
                animate={{
                  opacity: dragActive ? 0.3 : 0.1,
                }}
                transition={{ duration: 0.3 }}
              />

              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx,.json,.xml,.txt"
                onChange={handleFileChange}
                className="hidden"
              />

              <motion.button
                onClick={() => fileInputRef.current?.click()}
                className="w-full text-center relative z-10"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <motion.div
                  animate={{
                    y: dragActive ? -10 : 0,
                    scale: dragActive ? 1.1 : 1,
                  }}
                  transition={{ duration: 0.3 }}
                >
                  <Upload className={`w-16 h-16 mx-auto mb-4 transition-colors duration-300 ${dragActive ? 'text-primary-300' : 'text-primary-400'
                    }`} />
                </motion.div>

                <p className="text-xl font-semibold text-white mb-2">
                  {file ? file.name : 'Click to upload or drag and drop'}
                </p>
                <p className="text-sm text-gray-300">
                  CSV, XLSX, JSON, XML, TXT, or PDF files
                </p>

                {file && (
                  <motion.div
                    className="mt-4 flex items-center justify-center gap-2 text-green-400"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  >
                    <CheckCircle className="w-5 h-5" />
                    <span className="font-medium">File loaded successfully!</span>
                  </motion.div>
                )}
              </motion.button>

              {/* Upload Progress */}
              <AnimatePresence>
                {loading && uploadProgress > 0 && (
                  <motion.div
                    className="mt-6"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <ProgressBar progress={uploadProgress} color="primary" />
                    <p className="text-center text-sm text-gray-200 mt-2">
                      Loading file... {uploadProgress}%
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          </motion.div>

          {/* Error State */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <Card className="p-6 border-2 border-red-500/30 bg-red-900/30 backdrop-blur-sm">
                  <div className="flex items-start gap-4">
                    <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="font-semibold text-red-300 mb-1">Error</h3>
                      <p className="text-red-200/90">{error}</p>
                      <p className="text-sm text-red-300/70 mt-2">Make sure the ML server is running: python ml_server.py</p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* File Preview */}
          <AnimatePresence>
            {file && columns.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.5 }}
              >
                <Card className="p-6 md:p-8 bg-primary-900/40 backdrop-blur-sm border border-primary-500/20">
                  <h3 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                    <Eye className="w-6 h-6 text-primary-400" />
                    Preview
                  </h3>

                  {/* Column Selection */}
                  <div className="mb-6">
                    <label className="block text-sm font-semibold text-gray-200 mb-3">
                      Select Column with Headlines
                    </label>
                    <motion.select
                      value={columnName}
                      onChange={(e) => setColumnName(e.target.value)}
                      className="w-full px-4 py-3 bg-primary-800/50 border border-primary-600/40 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                      whileFocus={{ scale: 1.01 }}
                    >
                      {columns.map(col => (
                        <option key={col} value={col} className="bg-primary-900">{col}</option>
                      ))}
                    </motion.select>
                  </div>

                  {/* Preview Table */}
                  <div className="overflow-x-auto rounded-lg border border-primary-600/30">
                    <table className="w-full">
                      <thead className="bg-primary-800/50">
                        <tr>
                          {columns.map((col, idx) => (
                            <th key={idx} className="px-4 py-3 text-left text-sm font-semibold text-white">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {preview.map((row, idx) => (
                          <motion.tr
                            key={idx}
                            className="border-t border-primary-700/30 hover:bg-primary-800/30 transition-colors"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.05 }}
                          >
                            {row.map((cell, cellIdx) => (
                              <td key={cellIdx} className="px-4 py-3 text-sm text-gray-200">
                                {cell}
                              </td>
                            ))}
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Process Button */}
                  <motion.div
                    className="mt-6"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    <Button
                      onClick={handleProcess}
                      disabled={processing}
                      className="w-full md:w-auto bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-600 hover:to-accent-600 text-white font-semibold py-3 px-8 rounded-lg shadow-colored-teal hover:shadow-glow-teal transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {processing ? (
                        <>
                          <Loader className="w-5 h-5 mr-2 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-5 h-5 mr-2" />
                          Process Batch
                        </>
                      )}
                    </Button>
                  </motion.div>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results Section */}
          <AnimatePresence>
            {results && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.6 }}
              >
                <Card className="p-6 md:p-8 bg-primary-900/40 backdrop-blur-sm border border-primary-500/20">
                  <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
                    <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                      <CheckCircle className="w-6 h-6 text-green-400" />
                      Results ({results.results.length} headlines)
                    </h3>
                    <Button
                      onClick={handleDownload}
                      variant="outline"
                      className="w-full md:w-auto border-primary-500/40 text-white hover:bg-primary-800/50"
                    >
                      <Download className="w-5 h-5 mr-2" />
                      Download CSV
                    </Button>
                  </div>

                  <div className="overflow-x-auto rounded-lg border border-primary-600/30">
                    <table className="w-full">
                      <thead className="bg-primary-800/50">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-white">Headline</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-white">Classification</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-white">Sentiment</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-white">Key Words</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.results.map((row, idx) => (
                          <motion.tr
                            key={idx}
                            className="border-t border-primary-700/30 hover:bg-primary-800/30 transition-colors"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.02 }}
                          >
                            <td className="px-4 py-3 text-sm text-gray-200">{row.headline}</td>

                            <td className="px-4 py-3">
                              <div className="flex flex-col gap-1">
                                <Badge variant={row.is_clickbait ? 'danger' : 'success'}>
                                  {row.is_clickbait ? 'Clickbait' : 'Non-Clickbait'}
                                </Badge>
                                <span className="text-xs text-gray-400 ml-1">
                                  {(row.confidence * 100).toFixed(0)}%
                                </span>
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              {row.sentiment ? (
                                <Badge variant={
                                  row.sentiment?.toLowerCase() === 'positive' ? 'success' :
                                    row.sentiment?.toLowerCase() === 'negative' ? 'danger' : 'neutral'
                                }>
                                  {row.sentiment || 'Neutral'}
                                </Badge>
                              ) : (
                                <span className="text-xs text-gray-500">-</span>
                              )}
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex flex-wrap gap-1">
                                {row.highlighted_words && row.highlighted_words.length > 0 ? (
                                  row.highlighted_words.map((word, wIdx) => (
                                    <span key={wIdx} className="px-2 py-0.5 text-xs rounded-md bg-white/10 text-gray-300 border border-white/5">
                                      {word}
                                    </span>
                                  ))
                                ) : (
                                  <span className="text-xs text-gray-500">-</span>
                                )}
                              </div>
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Summary Sidebar */}
        <motion.div
          className="space-y-6"
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {/* File Info Card */}
          <Card className="p-6 bg-primary-900/40 backdrop-blur-sm border border-primary-500/20">
            <h3 className="text-lg font-semibold text-white mb-4">Summary</h3>

            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-300 mb-1">File</p>
                <p className="font-medium text-white">
                  {file ? file.name : 'No file selected'}
                </p>
              </div>

              <div>
                <p className="text-sm text-purple-300 mb-1">File Size</p>
                <p className="font-medium text-purple-100">
                  {file ? `${(file.size / 1024).toFixed(2)} KB` : '0 KB'}
                </p>
              </div>

              <div>
                <p className="text-sm text-purple-300 mb-1">Rows in Preview</p>
                <p className="font-medium text-purple-100">{preview.length}</p>
              </div>

              <div>
                <p className="text-sm text-purple-300 mb-1">Columns</p>
                <p className="font-medium text-purple-100">{columns.length}</p>
              </div>
            </div>
          </Card>

          {/* Stats Card */}
          <AnimatePresence>
            {results && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.4 }}
              >
                <Card className="p-6 bg-gradient-to-br from-primary-900/60 to-accent-900/40 backdrop-blur-sm border border-primary-500/30">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-primary-400" />
                    Analysis Stats
                  </h3>

                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Total Headlines</span>
                      <motion.span
                        className="text-2xl font-bold text-white"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      >
                        {results.results.length}
                      </motion.span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Clickbait</span>
                      <motion.span
                        className="text-xl font-bold text-red-400"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 300, delay: 0.1 }}
                      >
                        {stats.clickbait}
                      </motion.span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Non-Clickbait</span>
                      <motion.span
                        className="text-xl font-bold text-green-400"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 300, delay: 0.2 }}
                      >
                        {stats.nonClickbait}
                      </motion.span>
                    </div>

                    <div className="pt-4 border-t border-primary-700/40">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-gray-300">Clickbait Rate</span>
                        <span className="text-lg font-bold text-primary-300">{stats.percentage}%</span>
                      </div>
                      <ProgressBar progress={parseFloat(stats.percentage)} color="primary" />
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </motion.main>
  );
}
