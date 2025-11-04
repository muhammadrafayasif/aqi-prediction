import { useEffect, useState } from "react";

function App() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchPredictions() {
      try {
        const response = await fetch("http://127.0.0.1:8000/predict");
        if (!response.ok) {
          throw new Error("Failed to fetch predictions");
        }
        const data = await response.json();
        setPredictions(data.predictions || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchPredictions();
  }, []);

  if (loading) {
    return <div className="center-text">Loading...</div>;
  }

  if (error) {
    return <div className="center-text error-text">{error}</div>;
  }

  return (
    <div className="app">
      <h1>Air Quality Predictions</h1>
      <div className="card-container">
        {predictions.map((item, index) => (
          <div key={index} className="card">
            <p className="timestamp">
              {new Date(item.timestamp).toLocaleString()}
            </p>
            <p className="aqi">AQI: {item.predicted_aqi.toFixed(2)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;