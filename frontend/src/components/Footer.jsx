import { Link } from 'react-router-dom';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gray-900 text-gray-300 mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* About */}
          <div>
            <h3 className="text-white font-bold text-lg mb-4">DeClickify</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Advanced deep learning system for detecting clickbait headlines and analyzing sentiment with high accuracy.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-white font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm">
              <li><Link to="/" className="text-gray-400 hover:text-blue-400 transition font-medium">Home</Link></li>
              <li><Link to="/analyze" className="text-gray-400 hover:text-blue-400 transition font-medium">Analyze Headline</Link></li>
              <li><Link to="/batch" className="text-gray-400 hover:text-blue-400 transition font-medium">Batch Upload</Link></li>
              <li><Link to="/dashboard" className="text-gray-400 hover:text-blue-400 transition font-medium">Dashboard</Link></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-white font-semibold mb-4">Features & Use Cases</h4>
            <ul className="space-y-2 text-sm">
              <li><Link to="/about" className="text-gray-400 hover:text-blue-400 transition font-medium">How It Works</Link></li>
              <li><a href="mailto:contact@declickify.com" className="text-gray-400 hover:text-blue-400 transition font-medium">Contact Support</a></li>
              <li><Link to="/" className="text-gray-400 hover:text-blue-400 transition font-medium">View Use Cases</Link></li>
              <li><Link to="/batch" className="text-gray-400 hover:text-blue-400 transition font-medium">Bulk Processing</Link></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm">
              © {currentYear} DeClickify. All rights reserved.
            </p>
            <p className="text-gray-400 text-sm mt-4 md:mt-0">
              Built with React & Tailwind CSS • Powered by Deep Learning
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
