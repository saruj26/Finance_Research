import React from "react";
import "./PredictionIndicator.css";

type PredictionType = "up" | "down" | null;

interface PredictionResult {
  prediction: PredictionType;
  probability?: number;
  confidence?: number;
  probabilities?: {
    up: number;
    down: number;
  };
}

interface PredictionIndicatorProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

const PredictionIndicator: React.FC<PredictionIndicatorProps> = ({
  prediction,
  isLoading,
}) => {
  const predictionType = prediction?.prediction || null;

  return (
    <div className="prediction-section animate-fade-in-up">
      <h2 className="prediction-title">Market Movement Prediction</h2>

      {isLoading && (
        <div className="loading-indicator">
          <div className="loading-spinner"></div>
          <p>Analyzing article with AI models...</p>
        </div>
      )}

      {/* {!isLoading && prediction && (
        <div className="prediction-result">
          <div className="confidence-display">
            <span className="confidence-label">Confidence:</span>
            <span className="confidence-value">
              {(prediction.confidence! * 100).toFixed(1)}%
            </span>
          </div>
          <div className="probabilities">
            <div className="prob-bar">
              <span>
                UP: {(prediction.probabilities!.up * 100).toFixed(1)}%
              </span>
              <div className="bar">
                <div
                  className="bar-fill bar-up"
                  style={{ width: `${prediction.probabilities!.up * 100}%` }}
                ></div>
              </div>
            </div>
            <div className="prob-bar">
              <span>
                DOWN: {(prediction.probabilities!.down * 100).toFixed(1)}%
              </span>
              <div className="bar">
                <div
                  className="bar-fill bar-down"
                  style={{ width: `${prediction.probabilities!.down * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      )} */}

      <div className="bulbs-container">
        <div
          className={`bulb-wrapper ${predictionType === "up" ? "active" : "inactive"}`}
        >
          <div
            className={`bulb bulb-green ${predictionType === "up" ? "glow-green" : ""}`}
          >
            <div className="bulb-inner">
              <svg
                width="80"
                height="80"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="18 15 12 9 6 15" />
              </svg>
            </div>
          </div>
          <p className="bulb-label">Index Movement: UP</p>
          <div className="bulb-status">
            {predictionType === "up" && (
              <span className="status-badge status-active">ACTIVE</span>
            )}
          </div>
        </div>

        <div
          className={`bulb-wrapper ${predictionType === "down" ? "active" : "inactive"}`}
        >
          <div
            className={`bulb bulb-red ${predictionType === "down" ? "glow-red" : ""}`}
          >
            <div className="bulb-inner">
              <svg
                width="80"
                height="80"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </div>
          </div>
          <p className="bulb-label">Index Movement: DOWN</p>
          <div className="bulb-status">
            {predictionType === "down" && (
              <span className="status-badge status-active">ACTIVE</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionIndicator;
