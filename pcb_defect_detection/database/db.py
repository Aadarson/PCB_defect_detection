import sqlite3
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, 'defects.db')
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import api_logger

def init_db():
    """Initializes the SQLite database and creates the predictions table if it doesn't exist."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        api_logger.error(f"Database Initialization Error: {e}")
    finally:
        if conn:
            conn.close()

def log_prediction(filename: str, defect_type: str, confidence: float):
    """Logs a single defect prediction into the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (filename, defect_type, confidence)
            VALUES (?, ?, ?)
        ''', (filename, defect_type, confidence))
        conn.commit()
    except sqlite3.Error as e:
        api_logger.error(f"Database Insertion Error: {e}")
    finally:
        if conn:
            conn.close()

def get_stats():
    """Fetches defect statistics from the database for the dashboard."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total predictions made (not just images, total boxes)
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        # Average Confidence
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_resp = cursor.fetchone()[0]
        avg_confidence = round(avg_resp, 4) if avg_resp else 0.0
        
        # Counts per defect type
        cursor.execute('SELECT defect_type, COUNT(*) as count FROM predictions GROUP BY defect_type ORDER BY count DESC')
        defect_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Most common defect
        most_common = list(defect_counts.keys())[0] if defect_counts else "None"
        
        # Recent history
        cursor.execute('SELECT filename, defect_type, confidence, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 10')
        recent_history = [
            {"filename": r[0], "defect_type": r[1], "confidence": round(r[2], 2), "timestamp": r[3]} 
            for r in cursor.fetchall()
        ]
        
        return {
            "total_predictions": total_predictions,
            "average_confidence": avg_confidence,
            "defect_counts": defect_counts,
            "most_common_defect": most_common,
            "recent_history": recent_history
        }
    except sqlite3.Error as e:
        api_logger.error(f"Database Fetch Error: {e}")
        return {}
    finally:
        if conn:
            conn.close()

# Auto-initialize on import
init_db()
