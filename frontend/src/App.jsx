import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  TimeScale
} from "chart.js";
import "chartjs-adapter-date-fns";
import "./App.css";

// Register Chart.js components
ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  TimeScale
);

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

  // ----------------------------
  // Chart.js configuration
  // ----------------------------
  const chartData = {
    labels: predictions.map((item) => new Date(item.timestamp)),
    datasets: [
      {
        label: "AQI Forecast (Next 72h)",
        data: predictions.map((item) => item.aqi),
        borderColor: "rgba(75,192,192,1)",
        backgroundColor: "rgba(75,192,192,0.2)",
        fill: true,
        tension: 0.3,
        pointRadius: 2
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    scales: {
      x: {
        type: "time",
        time: {
          unit: "hour",
          tooltipFormat: "MMM d, HH:mm",
        },
        title: {
          display: true,
          text: "Time",
        },
      },
      y: {
        title: {
          display: true,
          text: "Air Quality Index (AQI)",
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: "top",
      },
      tooltip: {
        callbacks: {
          label: (context) => `AQI: ${context.parsed.y.toFixed(2)}`,
        },
      },
    },
  };

  return (
    <div className="app">
      <h1>Air Quality Predictions</h1>

      {/* Chart Section */}
      <div className="chart-container">
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* Cards Section */}
      <div className="card-container">
        {predictions.map((item, index) => (
          <div key={index} className="card">
            <p className="timestamp">
              {new Date(item.timestamp).toLocaleString()}
            </p>
            <p
              className={`aqi ${
                item.aqi < 50
                  ? "good"
                  : item.aqi < 100
                  ? "moderate"
                  : item.aqi < 150
                  ? "unhealthy"
                  : "hazardous"
              }`}
            >
              AQI: {item.aqi.toFixed(2)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
