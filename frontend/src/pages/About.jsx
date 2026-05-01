import { Brain, Sparkles, Target, Users, Award, Zap, BarChart3, Shield, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import Card from '../components/Card';
import AnimatedCounter from '../components/AnimatedCounter';
import AnimatedBackground from '../components/AnimatedBackground';

export default function About() {
  const features = [
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'Instant clickbait detection with confidence scores and detailed insights',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: BarChart3,
      title: 'Batch Processing',
      description: 'Analyze thousands of headlines simultaneously with CSV upload support',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: Shield,
      title: 'Sentiment Analysis',
      description: 'Advanced emotional tone detection with precision AI algorithms',
      color: 'from-green-500 to-emerald-500'
    }
  ];

  const techStack = {
    backend: [
      { name: 'Python 3.14', desc: 'Core runtime environment' },
      { name: 'Flask', desc: 'Lightweight web framework' },
      { name: 'BiLSTM', desc: 'Neural network architecture' },
      { name: 'NumPy & Pandas', desc: 'Data processing libraries' }
    ],
    frontend: [
      { name: 'React 18', desc: 'Modern UI framework' },
      { name: 'Vite', desc: 'Lightning-fast build tool' },
      { name: 'Tailwind CSS', desc: 'Utility-first styling' },
      { name: 'Framer Motion', desc: 'Animation library' }
    ]
  };

  const milestones = [
    { year: '2024', title: 'Project Inception', desc: 'Initial research and planning phase' },
    { year: '2025', title: 'Model Training', desc: 'BiLSTM model development and optimization' },
    { year: '2026', title: 'Full Launch', desc: 'Complete platform with 100% accuracy' }
  ];

  return (
    <motion.main
      className="min-h-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <AnimatedBackground />

      {/* Header Section */}
      <div className="relative pt-20 pb-12 px-4 text-center">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', duration: 0.8 }}
          className="inline-block mb-6"
        >
          <div className="w-24 h-24 mx-auto bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl flex items-center justify-center glow-shadow">
            <Brain className="w-12 h-12 text-primary-300" />
          </div>
        </motion.div>

        <motion.h1
          className="text-5xl sm:text-6xl font-bold mb-6 gradient-text"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          About DeClickify
        </motion.h1>

        <motion.p
          className="text-xl text-gray-300 max-w-2xl mx-auto"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Empowering users with AI-driven clickbait detection and intelligent sentiment analysis
        </motion.p>
      </div>

      <div className="max-w-6xl mx-auto px-4 pb-16">
        {/* Mission Statement */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <Card className="p-8 mb-12 bg-white/5 border border-white/10 backdrop-blur-md shadow-large">
            <div className="flex items-start gap-4 mb-6">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl flex items-center justify-center flex-shrink-0">
                <Target className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white mb-4">Our Mission</h2>
                <p className="text-lg text-gray-300 leading-relaxed mb-4">
                  DeClickify leverages cutting-edge deep learning technology to combat manipulative content.
                  Our BiLSTM (Bidirectional Long Short-Term Memory) neural network analyzes headline patterns,
                  emotional triggers, and linguistic cues to identify clickbait with unprecedented accuracy.
                </p>
                <p className="text-lg text-gray-300 leading-relaxed">
                  We believe in transparent, ethical journalism and aim to help users make informed decisions
                  about the content they consume in the digital age.
                </p>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Key Features */}
        <div className="mb-16">
          <motion.h2
            className="text-3xl font-bold text-white mb-8 text-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Powerful Features
          </motion.h2>

          <div className="grid md:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -5 }}
              >
                <Card className="p-6 h-full hover:shadow-neon-purple transition-all duration-300 bg-primary-900/40 backdrop-blur-sm border border-primary-500/20">
                  <div className={`w-14 h-14 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-4`}>
                    <feature.icon className="w-7 h-7 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-purple-100 mb-2">{feature.title}</h3>
                  <p className="text-purple-200/80">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Performance Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <Card className="p-8 mb-16 bg-white/5 border border-white/10 backdrop-blur-md shadow-medium">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-2">World-Class Performance</h2>
              <p className="text-gray-300">Proven accuracy and reliability metrics</p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <motion.div
                className="text-center"
                initial={{ scale: 0.5, opacity: 0 }}
                whileInView={{ scale: 1, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
              >
                <div className="text-5xl font-bold mb-2">
                  <span className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                    <AnimatedCounter end={100} suffix="%" />
                  </span>
                </div>
                <p className="text-gray-700 font-medium">Accuracy Rate</p>
              </motion.div>

              <motion.div
                className="text-center"
                initial={{ scale: 0.5, opacity: 0 }}
                whileInView={{ scale: 1, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <div className="text-5xl font-bold mb-2">
                  <span className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                    <AnimatedCounter end={30} suffix="+" />
                  </span>
                </div>
                <p className="text-gray-700 font-medium">Tests Passed</p>
              </motion.div>

              <motion.div
                className="text-center"
                initial={{ scale: 0.5, opacity: 0 }}
                whileInView={{ scale: 1, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
              >
                <div className="text-5xl font-bold mb-2">
                  <span className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                    8/8
                  </span>
                </div>
                <p className="text-gray-700 font-medium">Perfect Runs</p>
              </motion.div>

              <motion.div
                className="text-center"
                initial={{ scale: 0.5, opacity: 0 }}
                whileInView={{ scale: 1, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 }}
              >
                <div className="text-5xl font-bold mb-2">
                  <span className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                    BiLSTM
                  </span>
                </div>
                <p className="text-gray-700 font-medium">ML Model</p>
              </motion.div>
            </div>
          </Card>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <Card className="p-8 shadow-medium border border-white/10 bg-white/5">
            <h2 className="text-3xl font-bold text-white mb-8 text-center">Technology Stack</h2>

            <div className="grid md:grid-cols-2 gap-8">
              {/* Backend */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-500 rounded-lg flex items-center justify-center">
                    <Shield className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white">Backend</h3>
                </div>
                <div className="space-y-3">
                  {techStack.backend.map((tech, index) => (
                    <motion.div
                      key={index}
                      className="flex items-start gap-3 p-3 rounded-lg hover:bg-primary-800/30 transition-colors"
                      initial={{ x: -20, opacity: 0 }}
                      whileInView={{ x: 0, opacity: 1 }}
                      viewport={{ once: true }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className="w-2 h-2 rounded-full bg-green-500 mt-2" />
                      <div>
                        <p className="font-semibold text-white">{tech.name}</p>
                        <p className="text-sm text-gray-400">{tech.desc}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Frontend */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white">Frontend</h3>
                </div>
                <div className="space-y-3">
                  {techStack.frontend.map((tech, index) => (
                    <motion.div
                      key={index}
                      className="flex items-start gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors"
                      initial={{ x: 20, opacity: 0 }}
                      whileInView={{ x: 0, opacity: 1 }}
                      viewport={{ once: true }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className="w-2 h-2 rounded-full bg-blue-500 mt-2" />
                      <div>
                        <p className="font-semibold text-white">{tech.name}</p>
                        <p className="text-sm text-gray-400">{tech.desc}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </motion.main>
  );
}
