import { useState } from 'react';
import { AlertCircle, CheckCircle, Loader, TrendingUp, Zap, Copy } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Card from '../components/Card';
import Button from '../components/Button';
import Badge from '../components/Badge';
import LoadingSkeleton from '../components/LoadingSkeleton';
import AnimatedBackground from '../components/AnimatedBackground';
import { analyzeHeadline } from '../services/api';
import { showSuccess, showError } from '../utils/toast';

export default function Analyze() {
  const [headline, setHeadline] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!headline.trim()) {
      showError('Please enter a headline to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeHeadline(headline);
      setResult(data);
      showSuccess('Analysis complete!');
    } catch (err) {
      const errorMessage = err.message || 'Failed to analyze headline';
      setError(errorMessage);
      showError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyHeadline = () => {
    navigator.clipboard.writeText(headline);
    showSuccess('Headline copied to clipboard!');
  };

  return (
    <motion.main
      className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <AnimatedBackground />
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold gradient-text mb-2">Analyze Headline</h1>
        <p className="text-lg text-gray-300">
          Enter a headline to detect clickbait patterns and analyze sentiment
        </p>
      </div>

      {/* Model Status Indicator */}
      {result && (
        <motion.div
          className={`p-4 rounded-lg mb-8 flex items-center gap-2 ${result.models_available
            ? 'bg-green-500/10 border border-green-500/20 text-green-300'
            : 'bg-yellow-500/10 border border-yellow-500/20 text-yellow-300'
            }`}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          {result.models_available ? (
            <>
              <Zap className="w-5 h-5" />
              <span className="text-sm font-medium">BiLSTM Model Active</span>
            </>
          ) : (
            <>
              <AlertCircle className="w-5 h-5" />
              <span className="text-sm font-medium">Analysis Unavailable</span>
            </>
          )}
        </motion.div>
      )}

      {/* Input Section */}
      <Card className="p-8 mb-8 shadow-soft hover:shadow-medium transition-shadow duration-300">
        <form onSubmit={handleAnalyze}>
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-semibold text-white">
                Headline
              </label>
              {headline && (
                <button
                  type="button"
                  onClick={handleCopyHeadline}
                  className="flex items-center gap-2 text-sm text-primary-400 hover:text-primary-300 transition-colors"
                >
                  <Copy className="w-4 h-4" />
                  Copy
                </button>
              )}
            </div>
            <textarea
              value={headline}
              onChange={(e) => setHeadline(e.target.value)}
              placeholder="Enter the headline you want to analyze..."
              className="w-full px-4 py-3 bg-white/5 border-2 border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none transition-all duration-200"
              rows={4}
            />
            <div className="flex items-center justify-between mt-2">
              <p className={`text-sm font-medium transition-colors ${headline.length > 200 ? 'text-red-400' :
                headline.length > 100 ? 'text-yellow-400' :
                  'text-gray-400'
                }`}>
                {headline.length} characters
              </p>
              {headline.length > 200 && (
                <p className="text-xs text-red-400">Headlines are typically shorter</p>
              )}
            </div>
          </div>

          <Button
            type="submit"
            size="lg"
            isLoading={loading}
            disabled={!headline.trim() || loading}
            className="w-full sm:w-auto shadow-colored-blue hover:shadow-lg transition-all"
          >
            {loading ? 'Analyzing...' : 'Analyze Headline'}
          </Button>
        </form>
      </Card>

      {/* Error State */}
      <AnimatePresence>
        {error && !loading && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Card className="p-6 mb-8 border-2 border-red-200 bg-red-50">
              <div className="flex items-start gap-4">
                <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-red-900 mb-1">Error</h3>
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading State */}
      {loading && (
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Card className="p-8">
            <LoadingSkeleton type="text" width="w-48" height="h-6" className="mb-4" />
            <LoadingSkeleton type="text" count={2} className="mb-6" />
            <LoadingSkeleton type="text" width="w-32" height="h-10" />
          </Card>
          <Card className="p-8">
            <LoadingSkeleton type="text" width="w-48" height="h-6" className="mb-4" />
            <LoadingSkeleton type="text" count={2} />
          </Card>
          <Card className="p-8">
            <LoadingSkeleton type="text" width="w-48" height="h-6" className="mb-4" />
            <div className="flex flex-wrap gap-2">
              <LoadingSkeleton type="text" width="w-20" height="h-8" count={5} />
            </div>
          </Card>
        </motion.div>
      )}

      {/* Results */}
      <AnimatePresence>
        {result && !loading && (
          <motion.div
            className="space-y-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Clickbait Result */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card className="p-8 border-2 border-blue-100 hover:shadow-medium transition-shadow">
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-2">Clickbait Detection</h3>
                    <p className="text-gray-300">AI analysis result</p>
                  </div>
                  <motion.div
                    className="text-right"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.3, type: "spring" }}
                  >
                    {result.is_clickbait ? (
                      <AlertCircle className="w-8 h-8 text-red-600" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-green-600" />
                    )}
                  </motion.div>
                </div>

                <div className="flex items-center gap-3 mb-6">
                  <Badge
                    label={result.is_clickbait ? '⚠️ Clickbait Detected' : '✓ Legitimate Headline'}
                    variant={result.is_clickbait ? 'danger' : 'success'}
                  />
                </div>

                {/* Confidence Progress Bar */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-300">Confidence</span>
                    <span className="text-sm font-semibold text-white">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                    <motion.div
                      className={`h-full rounded-full ${result.is_clickbait
                        ? 'bg-gradient-to-r from-red-500 to-red-600'
                        : 'bg-gradient-to-r from-green-500 to-green-600'
                        }`}
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence * 100}%` }}
                      transition={{ duration: 1, delay: 0.4, ease: "easeOut" }}
                    />
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* Sentiment Result */}
            {/* Sentiment Result */}
            {result.sentiment && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <Card className="p-8 border-2 border-purple-100 hover:shadow-medium transition-shadow">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-2">Sentiment Analysis</h3>
                      <p className="text-gray-300">Emotional tone detection</p>
                    </div>
                    <motion.div
                      initial={{ scale: 0, rotate: -180 }}
                      animate={{ scale: 1, rotate: 0 }}
                      transition={{ delay: 0.4, type: "spring" }}
                    >
                      <TrendingUp className="w-8 h-8 text-purple-400" />
                    </motion.div>
                  </div>

                  <div className="flex items-center gap-3 mb-6">
                    <Badge
                      label={result.sentiment}
                      variant={
                        result.sentiment === 'Positive'
                          ? 'success'
                          : result.sentiment === 'Negative'
                            ? 'danger'
                            : 'neutral'
                      }
                    />
                  </div>

                  {/* Sentiment Confidence */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-300">Confidence</span>
                      <span className="text-sm font-semibold text-white">
                        {(result.sentiment_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-purple-500 to-purple-600"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.sentiment_confidence * 100}%` }}
                        transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
                      />
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}

            {/* Highlighted Words */}
            {result.highlighted_words && result.highlighted_words.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="p-8 hover:shadow-medium transition-shadow">
                  <h3 className="text-lg font-semibold text-white mb-4">Key Words Contributing to Clickbait</h3>
                  <div className="flex flex-wrap gap-2">
                    {result.highlighted_words.map((word, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.4 + (idx * 0.05) }}
                        whileHover={{ scale: 1.05 }}
                      >
                        <Badge label={word} variant="primary" />
                      </motion.div>
                    ))}
                  </div>
                </Card>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty State */}
      {!result && !loading && (
        <Card className="p-12 text-center">
          <Loader className="w-12 h-12 text-gray-500 mx-auto mb-4 opacity-50" />
          <p className="text-gray-300">Enter a headline above and click "Analyze Headline" to see results</p>
        </Card>
      )}
    </motion.main>
  );
}
