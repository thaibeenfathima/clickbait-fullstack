import { BarChart3, Zap, TrendingUp, Shield, CheckCircle, Download, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import Button from '../components/Button';
import Card from '../components/Card';
import AnimatedCounter from '../components/AnimatedCounter';

export default function Home() {
  const features = [
    {
      icon: BarChart3,
      title: 'Accurate Detection',
      description: 'BiLSTM neural network trained on thousands of headlines detects clickbait patterns with high precision.',
      color: 'blue',
      delay: 0.1
    },
    {
      icon: Zap,
      title: 'Real-Time Analysis',
      description: 'Get instant results for single headlines or batch process thousands of articles in seconds.',
      color: 'purple',
      delay: 0.2
    },
    {
      icon: TrendingUp,
      title: 'Sentiment Insights',
      description: 'Analyze emotional tone alongside clickbait detection for comprehensive content understanding.',
      color: 'green',
      delay: 0.3
    },
    {
      icon: Shield,
      title: 'Keyword Highlighting',
      description: 'See exactly which words and phrases trigger clickbait detection for transparency.',
      color: 'red',
      delay: 0.4
    },
    {
      icon: CheckCircle,
      title: 'Easy Integration',
      description: 'Simple REST API and intuitive UI make it easy to integrate clickbait detection into your workflow.',
      color: 'yellow',
      delay: 0.5
    },
    {
      icon: Download,
      title: 'Export Results',
      description: 'Download detailed analysis results in CSV format for further processing and reporting.',
      color: 'indigo',
      delay: 0.6
    }
  ];

  const colorClasses = {
    blue: 'bg-blue-500/20 text-blue-300',
    purple: 'bg-purple-500/20 text-purple-300',
    green: 'bg-green-500/20 text-green-300',
    red: 'bg-red-500/20 text-red-300',
    yellow: 'bg-yellow-500/20 text-yellow-300',
    indigo: 'bg-indigo-500/20 text-indigo-300'
  };

  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-primary-600 via-primary-700 to-accent-700 text-white py-24 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            className="absolute -top-1/2 -left-1/4 w-96 h-96 bg-primary-400/20 rounded-full blur-3xl"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.5, 0.3]
            }}
            transition={{ duration: 8, repeat: Infinity }}
          />
          <motion.div
            className="absolute -bottom-1/2 -right-1/4 w-96 h-96 bg-accent-400/20 rounded-full blur-3xl"
            animate={{
              scale: [1.2, 1, 1.2],
              opacity: [0.3, 0.5, 0.3]
            }}
            transition={{ duration: 8, repeat: Infinity, delay: 1 }}
          />
        </div>

        <div className="relative max-w-5xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex items-center justify-center gap-2 mb-4">
              <Sparkles className="w-8 h-8 text-yellow-300" />
              <h1 className="text-5xl sm:text-6xl font-bold">DeClickify</h1>
              <Sparkles className="w-8 h-8 text-yellow-300" />
            </div>
          </motion.div>

          <motion.p
            className="text-xl sm:text-2xl text-primary-100 mb-8 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            Advanced AI-Powered Clickbait Detection & Sentiment Analysis Platform
          </motion.p>

          <motion.div
            className="flex justify-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Link to="/analyze">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  size="lg"
                  className="bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:from-primary-600 hover:to-primary-700 shadow-colored-teal hover:shadow-glow-teal transition-all px-10 py-4 text-lg font-bold border-2 border-white/20"
                >
                  Get Started →
                </Button>
              </motion.div>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Why Choose DeClickify */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-4xl font-bold text-white mb-4 text-center">
              Why Choose DeClickify?
            </h2>
            <p className="text-lg text-gray-300 text-center mb-16 max-w-2xl mx-auto">
              Powerful features designed to help you identify clickbait and analyze sentiment with confidence
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: feature.delay }}
                  whileHover={{ y: -5 }}
                >
                  <Card className="p-8 h-full hover:bg-white/5 border border-white/10 transition-all duration-300">
                    <div className={`w-12 h-12 ${colorClasses[feature.color]} rounded-lg flex items-center justify-center mb-4`}>
                      <Icon className="w-6 h-6" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
                    <p className="text-gray-400">
                      {feature.description}
                    </p>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white/5 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto">
          <motion.h2
            className="text-3xl font-bold text-white mb-12 text-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            Proven Performance
          </motion.h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <motion.div
              className="text-center"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="text-5xl font-bold mb-2">
                <span className="gradient-text">
                  <AnimatedCounter end={100} suffix="%" />
                </span>
              </div>
              <p className="text-gray-300 font-medium">Accuracy Rate</p>
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="text-5xl font-bold mb-2">
                <span className="gradient-text">
                  <AnimatedCounter end={30} suffix="+" />
                </span>
              </div>
              <p className="text-gray-300 font-medium">Headlines Tested</p>
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <div className="text-5xl font-bold mb-2">
                <span className="gradient-text">8/8</span>
              </div>
              <p className="text-gray-300 font-medium">Perfect Tests</p>
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <div className="text-5xl font-bold mb-2">
                <span className="gradient-text">BiLSTM</span>
              </div>
              <p className="text-gray-300 font-medium">Model Type</p>
            </motion.div>
          </div>
        </div>
      </section>
    </main>
  );
}
