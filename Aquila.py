import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, Shield, DollarSign, AlertTriangle, CheckCircle, XCircle, Activity, BarChart3, Zap, Lock, Unlock } from 'lucide-react';

const AquilaPremium = () => {
  const [step, setStep] = useState(1);
  const [score, setScore] = useState(3.0);
  const [ead, setEad] = useState(1000000);
  const [guarantees, setGuarantees] = useState(600000);
  const [rate, setRate] = useState(0.025);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [showDecision, setShowDecision] = useState(false);
  const [particles, setParticles] = useState([]);
  const canvasRef = useRef(null);

  // Animated background particles
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particlesArray = [];
    for (let i = 0; i < 50; i++) {
      particlesArray.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2 + 1,
        speedX: Math.random() * 0.5 - 0.25,
        speedY: Math.random() * 0.5 - 0.25,
        opacity: Math.random() * 0.5 + 0.2
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      particlesArray.forEach(particle => {
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(96, 165, 250, ${particle.opacity})`;
        ctx.fill();

        particle.x += particle.speedX;
        particle.y += particle.speedY;

        if (particle.x > canvas.width) particle.x = 0;
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.y > canvas.height) particle.y = 0;
        if (particle.y < 0) particle.y = canvas.height;
      });

      requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const calculateRisk = () => {
    setAnalyzing(true);
    setShowDecision(false);

    setTimeout(() => {
      const lambda1 = -Math.log(1 - 0.80);
      const lambda5 = -Math.log(1 - 0.05);
      const alpha = (score - 1.0) / 4.0;
      const lambda = Math.exp(Math.log(lambda1) + alpha * (Math.log(lambda5) - Math.log(lambda1)));
      
      const pd12 = 1.0 - Math.exp(-lambda);
      const lgd = 1.0 - Math.min(guarantees, ead) / ead;
      const ecl = lgd * ead;
      const annualRate = Math.pow(1 + rate, 12) - 1;
      const expectedReturn = annualRate * (1 - pd12) - lgd * pd12;

      setResults({
        pd12,
        lgd,
        ecl,
        expectedReturn,
        decision: expectedReturn >= 0.12
      });
      setAnalyzing(false);
      setTimeout(() => setShowDecision(true), 100);
    }, 2000);
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const getRiskColor = () => {
    if (score < 2.5) return 'from-red-500 to-red-600';
    if (score < 4.0) return 'from-yellow-500 to-orange-500';
    return 'from-green-500 to-emerald-600';
  };

  const getRiskLabel = () => {
    if (score < 2.5) return 'Alto Riesgo';
    if (score < 4.0) return 'Riesgo Medio';
    return 'Bajo Riesgo';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white relative overflow-hidden">
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none opacity-30" />
      
      {/* Gradient Orbs */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-700" />
      
      <div className="relative z-10 container mx-auto px-6 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-flex items-center gap-3 mb-4 px-6 py-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 backdrop-blur-xl rounded-full border border-blue-500/20">
            <Shield className="w-6 h-6 text-blue-400" />
            <span className="text-sm font-semibold tracking-wider">SISTEMA AQUILA v7.0</span>
          </div>
          <h1 className="text-6xl font-black mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Análisis de Riesgo Crediticio
          </h1>
          <p className="text-xl text-slate-400">Inteligencia Artificial para Decisiones Financieras</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-12 flex justify-center">
          <div className="flex items-center gap-4">
            {[1, 2, 3, 4, 5].map((s) => (
              <React.Fragment key={s}>
                <div className={`flex items-center justify-center w-12 h-12 rounded-full font-bold transition-all duration-500 ${
                  step >= s 
                    ? 'bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg shadow-blue-500/50 scale-110' 
                    : 'bg-slate-800/50 backdrop-blur-sm border border-slate-700'
                }`}>
                  {s}
                </div>
                {s < 5 && (
                  <div className={`w-16 h-1 rounded-full transition-all duration-500 ${
                    step > s ? 'bg-gradient-to-r from-blue-500 to-purple-600' : 'bg-slate-700'
                  }`} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Left Column - Inputs */}
          <div className="space-y-6">
            {/* Client Selection */}
            <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl rounded-2xl p-8 border border-slate-700/50 shadow-2xl hover:shadow-blue-500/10 transition-all duration-500 hover:scale-[1.02]">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl">
                  <BarChart3 className="w-6 h-6" />
                </div>
                <h3 className="text-2xl font-bold">Exposición al Riesgo</h3>
              </div>
              
              <div className="space-y-6">
                <div>
                  <label className="text-sm font-semibold text-slate-400 mb-2 block">EAD (Exposición)</label>
                  <div className="relative">
                    <DollarSign className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                    <input
                      type="number"
                      value={ead}
                      onChange={(e) => setEad(Number(e.target.value))}
                      className="w-full pl-12 pr-4 py-4 bg-slate-900/50 border border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all text-lg font-semibold"
                      step="10000"
                    />
                  </div>
                  <div className="mt-2 text-sm text-slate-400">{formatCurrency(ead)}</div>
                </div>

                <div>
                  <label className="text-sm font-semibold text-slate-400 mb-2 block">Garantías</label>
                  <div className="relative">
                    <Shield className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                    <input
                      type="number"
                      value={guarantees}
                      onChange={(e) => setGuarantees(Number(e.target.value))}
                      className="w-full pl-12 pr-4 py-4 bg-slate-900/50 border border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-lg font-semibold"
                      step="10000"
                    />
                  </div>
                  <div className="mt-2 text-sm text-slate-400">{formatCurrency(guarantees)}</div>
                </div>
              </div>
            </div>

            {/* Rate Configuration */}
            <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl rounded-2xl p-8 border border-slate-700/50 shadow-2xl hover:shadow-purple-500/10 transition-all duration-500 hover:scale-[1.02]">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl">
                  <Activity className="w-6 h-6" />
                </div>
                <h3 className="text-2xl font-bold">Tasa Compensatoria</h3>
              </div>
              
              <div>
                <label className="text-sm font-semibold text-slate-400 mb-2 block">Tasa Mensual</label>
                <input
                  type="number"
                  value={rate}
                  onChange={(e) => setRate(Number(e.target.value))}
                  className="w-full px-4 py-4 bg-slate-900/50 border border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-lg font-semibold"
                  step="0.001"
                  min="0"
                  max="0.2"
                />
                <div className="mt-4 flex items-center justify-between p-4 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-xl">
                  <span className="text-sm text-slate-400">Tasa Anual Equivalente</span>
                  <span className="text-2xl font-bold text-purple-400">
                    {formatPercent(Math.pow(1 + rate, 12) - 1)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Score & Risk */}
          <div className="space-y-6">
            {/* Risk Score */}
            <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl rounded-2xl p-8 border border-slate-700/50 shadow-2xl">
              <div className="flex items-center gap-3 mb-6">
                <div className={`p-3 bg-gradient-to-br ${getRiskColor()} rounded-xl`}>
                  <Zap className="w-6 h-6" />
                </div>
                <h3 className="text-2xl font-bold">Calificación de Riesgo</h3>
              </div>

              <div className="mb-8">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-5xl font-black">{score.toFixed(1)}</span>
                  <span className={`px-4 py-2 rounded-full text-sm font-bold bg-gradient-to-r ${getRiskColor()}`}>
                    {getRiskLabel()}
                  </span>
                </div>
                
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="0.1"
                  value={score}
                  onChange={(e) => setScore(Number(e.target.value))}
                  className="w-full h-3 rounded-full appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, 
                      rgb(239, 68, 68) 0%, 
                      rgb(251, 146, 60) 25%, 
                      rgb(234, 179, 8) 50%, 
                      rgb(132, 204, 22) 75%, 
                      rgb(34, 197, 94) 100%)`
                  }}
                />
                
                <div className="flex justify-between text-xs text-slate-500 mt-2">
                  <span>Alto Riesgo</span>
                  <span>Bajo Riesgo</span>
                </div>
              </div>

              {/* Risk Indicators */}
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: 'Liquidez', value: Math.min(95, 60 + score * 8), icon: TrendingUp },
                  { label: 'Solvencia', value: Math.min(90, 50 + score * 10), icon: Shield },
                  { label: 'Rentabilidad', value: Math.min(92, 55 + score * 9), icon: DollarSign }
                ].map((indicator, i) => (
                  <div key={i} className="text-center p-4 bg-slate-900/50 rounded-xl">
                    <indicator.icon className="w-5 h-5 mx-auto mb-2 text-blue-400" />
                    <div className="text-2xl font-bold mb-1">{indicator.value.toFixed(0)}%</div>
                    <div className="text-xs text-slate-400">{indicator.label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={calculateRisk}
              disabled={analyzing}
              className="w-full py-6 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-2xl font-bold text-xl shadow-2xl shadow-blue-500/50 hover:shadow-blue-500/80 transition-all duration-500 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 relative overflow-hidden group"
            >
              {analyzing ? (
                <>
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white" />
                  <span>Analizando...</span>
                </>
              ) : (
                <>
                  <Zap className="w-6 h-6" />
                  <span>EJECUTAR ANÁLISIS</span>
                </>
              )}
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-500" />
            </button>
          </div>
        </div>

        {/* Results Section */}
        {results && (
          <div className={`transition-all duration-700 ${showDecision ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            {/* Decision Banner */}
            <div className={`mb-8 p-12 rounded-3xl border-4 text-center relative overflow-hidden ${
              results.decision 
                ? 'bg-gradient-to-br from-green-500/20 to-emerald-500/20 border-green-500' 
                : 'bg-gradient-to-br from-red-500/20 to-rose-500/20 border-red-500'
            }`}>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent animate-pulse" />
              <div className="relative z-10">
                {results.decision ? (
                  <CheckCircle className="w-24 h-24 mx-auto mb-4 text-green-400 animate-bounce" />
                ) : (
                  <XCircle className="w-24 h-24 mx-auto mb-4 text-red-400 animate-bounce" />
                )}
                <h2 className="text-5xl font-black mb-4">
                  {results.decision ? '✅ CRÉDITO APROBADO' : '⛔ CRÉDITO RECHAZADO'}
                </h2>
                <p className="text-xl text-slate-300">
                  Retorno Esperado: <span className="font-bold text-2xl">{formatPercent(results.expectedReturn)}</span>
                  {results.decision ? ' (Supera umbral)' : ' (Por debajo del umbral)'}
                </p>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { 
                  label: 'Probabilidad de Default', 
                  value: formatPercent(results.pd12), 
                  icon: AlertTriangle,
                  color: 'from-red-500 to-orange-500',
                  description: '12 meses'
                },
                { 
                  label: 'Pérdida por Default', 
                  value: formatPercent(results.lgd), 
                  icon: TrendingDown,
                  color: 'from-orange-500 to-yellow-500',
                  description: 'LGD'
                },
                { 
                  label: 'Pérdida Esperada', 
                  value: formatCurrency(results.ecl), 
                  icon: DollarSign,
                  color: 'from-yellow-500 to-amber-500',
                  description: 'ECL'
                },
                { 
                  label: 'Retorno Esperado', 
                  value: formatPercent(results.expectedReturn), 
                  icon: results.expectedReturn >= 0.12 ? TrendingUp : TrendingDown,
                  color: results.expectedReturn >= 0.12 ? 'from-green-500 to-emerald-500' : 'from-red-500 to-rose-500',
                  description: 'Anual'
                }
              ].map((metric, i) => (
                <div 
                  key={i}
                  className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 hover:scale-105 transition-all duration-500 shadow-xl"
                  style={{ animationDelay: `${i * 100}ms` }}
                >
                  <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${metric.color} mb-4`}>
                    <metric.icon className="w-6 h-6" />
                  </div>
                  <div className="text-sm text-slate-400 mb-2">{metric.label}</div>
                  <div className="text-3xl font-black mb-1">{metric.value}</div>
                  <div className="text-xs text-slate-500">{metric.description}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="relative z-10 text-center py-8 text-slate-500 text-sm">
        <p>© 2025 Juan José Mostajo León · Aquila Risk Analysis System v7.0</p>
      </div>
    </div>
  );
};

export default AquilaPremium;
