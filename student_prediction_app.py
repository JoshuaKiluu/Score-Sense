"""
Student Grade Prediction System – SMA 390 Scientific Computing
Tkinter GUI Application

INSTRUCTIONS TO RUN:
  1. First run student_grade_prediction.ipynb fully to train the model.
  2. Then run:  python student_prediction_app.py
  (Both files must be in the same folder as student_performance.csv)

SMA 390 Concepts Demonstrated:
  - Topic 2: Data types (int, float, str, dict, list)
  - Topic 3: Control structures (if/elif/else, for loops)
  - Topic 4: Libraries (tkinter, numpy, pandas, scikit-learn)
  - Topic 5: Modular functions / subroutines
  - Topic 7: Real-world Python application
"""

# ─────────────────────────────────────────────
# IMPORTS  (Topic 4 – Sub-routine Libraries)
# ─────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np                          # ndarray (Topic 2)
import pandas as pd                         # DataFrame (Topic 2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONSTANTS  (Data types – Topic 2)
# ─────────────────────────────────────────────
CSV_PATH:      str   = 'student_performance.csv'
FEATURES:      list  = ['weekly_self_study_hours', 'attendance_percentage',
                         'class_participation', 'total_score']
GRADE_ORDER:   list  = ['A', 'B', 'C', 'D', 'F']
GRADE_COLORS:  dict  = {'A': '#2ECC71', 'B': '#3498DB',
                        'C': '#F39C12', 'D': '#E67E22', 'F': '#E74C3C'}
GRADE_LABELS:  dict  = {
    'A': 'Excellent  (75 – 100%)',
    'B': 'Good       (60 – 74%)',
    'C': 'Average    (50 – 59%)',
    'D': 'Below Avg  (40 – 49%)',
    'F': 'Fail       (0  – 39%)'
}
N_ROWS:        int   = 50000    # Sample size for quick training
TEST_SIZE:     float = 0.20
RANDOM_STATE:  int   = 42


# ─────────────────────────────────────────────
# SUBROUTINE: train_model()  (Topic 4 + 5)
# ─────────────────────────────────────────────
def train_model() -> tuple:
    """
    Load data, train Random Forest, return (model, encoder).
    This is a reusable subroutine (Topic 5 – Modular Code).
    """
    print('[INFO] Loading dataset...')
    df = pd.read_csv(CSV_PATH, nrows=N_ROWS)

    # Data preparation
    df = df.drop(columns=['student_id']).drop_duplicates().dropna()
    le = LabelEncoder()
    y  = le.fit_transform(df['grade'])
    X  = df[FEATURES]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    print('[INFO] Model trained successfully.')
    return model, le


# ─────────────────────────────────────────────
# SUBROUTINE: predict_grade()  (Topic 4 + 5)
# ─────────────────────────────────────────────
def predict_grade(model: RandomForestClassifier,
                  le: LabelEncoder,
                  values: list) -> dict:
    """
    Predict grade from a list of 4 numeric input values.
    Returns dict with grade, confidence, and probabilities.
    """
    study, attend, part, score = values  # Tuple unpacking (Topic 2)

    # Input validation (if/elif/else – Topic 3)
    if not (0 <= attend <= 100):
        raise ValueError('Attendance must be between 0 and 100.')
    elif not (0 <= part <= 10):
        raise ValueError('Participation score must be 0–10.')
    elif not (0 <= score <= 100):
        raise ValueError('Total score must be between 0 and 100.')
    elif study < 0:
        raise ValueError('Study hours cannot be negative.')

    input_arr: np.ndarray = np.array([[study, attend, part, score]])
    input_df  = pd.DataFrame(input_arr, columns=FEATURES)

    prediction    = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    grade: str    = le.inverse_transform([prediction])[0]

    prob_dict: dict = {
        g: round(float(p) * 100, 1)
        for g, p in zip(le.classes_, probabilities)
    }

    return {
        'grade':        grade,
        'confidence':   prob_dict[grade],
        'probabilities': prob_dict,
        'description':  GRADE_LABELS[grade]
    }


# ─────────────────────────────────────────────
# GUI CLASS  (Topic 7 – Application)
# ─────────────────────────────────────────────
class StudentPredictorApp:
    """Main Tkinter application window."""

    def __init__(self, root: tk.Tk, model, encoder):
        self.root    = root
        self.model   = model
        self.encoder = encoder
        self._build_ui()

    def _build_ui(self):
        """Build all UI widgets."""
        self.root.title('Student Grade Prediction System – SMA 390')
        self.root.geometry('600x680')
        self.root.resizable(False, False)
        self.root.configure(bg='#F2F4F7')

        # ── Title bar ──
        title_frame = tk.Frame(self.root, bg='#1F4E79', height=55)
        title_frame.pack(fill='x')
        tk.Label(title_frame,
                 text='🎓  Student Grade Prediction System',
                 font=('Arial', 14, 'bold'),
                 bg='#1F4E79', fg='white').pack(pady=14)

        # ── Subtitle ──
        tk.Label(self.root, text='SMA 390 – Scientific Computing  |  Random Forest Classifier',
                 font=('Arial', 9, 'italic'), bg='#F2F4F7', fg='#777777').pack(pady=(6, 0))

        # ── Input card ──
        card = tk.Frame(self.root, bg='white', bd=0,
                        highlightbackground='#DDDDDD', highlightthickness=1)
        card.pack(fill='x', padx=20, pady=12, ipady=8)

        card_header = tk.Frame(card, bg='#2E75B6')
        card_header.pack(fill='x')
        tk.Label(card_header, text='Enter Student Information',
                 font=('Arial', 12, 'bold'), bg='#2E75B6', fg='white').pack(pady=10)

        # Input rows: (label, variable, placeholder, min, max)
        fields: list = [
            ('Weekly Self-Study Hours (hrs):',  tk.StringVar(), '0 – 50',  '0',  '50'),
            ('Attendance Percentage (%):',       tk.StringVar(), '0 – 100', '0',  '100'),
            ('Class Participation Score (1–10):', tk.StringVar(), '1 – 10', '0',  '10'),
            ('Total Score (0–100):',             tk.StringVar(), '0 – 100', '0',  '100'),
        ]
        self.vars:    list = []
        self.entries: list = []

        for label_text, var, hint, *_ in fields:
            row_frame = tk.Frame(card, bg='white')
            row_frame.pack(fill='x', padx=20, pady=6)
            tk.Label(row_frame, text=label_text,
                     font=('Arial', 10, 'bold'),
                     bg='white', fg='#1F4E79', anchor='w').pack(fill='x')
            entry = tk.Entry(row_frame, textvariable=var,
                             font=('Arial', 11), bd=1,
                             highlightbackground='#2E75B6',
                             highlightthickness=1,
                             relief='flat', bg='#F8F9FA')
            entry.pack(fill='x', ipady=5)
            tk.Label(row_frame, text=f'Range: {hint}',
                     font=('Arial', 8), bg='white', fg='#AAAAAA',
                     anchor='w').pack(fill='x')
            self.vars.append(var)
            self.entries.append(entry)

        # ── Buttons ──
        btn_frame = tk.Frame(self.root, bg='#F2F4F7')
        btn_frame.pack(fill='x', padx=20, pady=(0, 6))

        tk.Button(btn_frame, text='  🔍  PREDICT GRADE  ',
                  font=('Arial', 12, 'bold'),
                  bg='#2E75B6', fg='white',
                  activebackground='#1F4E79',
                  relief='flat', cursor='hand2',
                  command=self._on_predict).pack(side='left', expand=True,
                                                  fill='x', ipady=10, padx=(0, 5))

        tk.Button(btn_frame, text='  ↺  Reset  ',
                  font=('Arial', 11),
                  bg='#E74C3C', fg='white',
                  activebackground='#C0392B',
                  relief='flat', cursor='hand2',
                  command=self._on_reset).pack(side='right', ipady=10, padx=(5, 0))

        # ── Result card ──
        result_card = tk.Frame(self.root, bg='white', bd=0,
                               highlightbackground='#DDDDDD',
                               highlightthickness=1)
        result_card.pack(fill='x', padx=20, pady=(0, 10), ipady=4)

        self.result_header = tk.Frame(result_card, bg='#AAAAAA')
        self.result_header.pack(fill='x')
        tk.Label(self.result_header, text='Prediction Result',
                 font=('Arial', 11, 'bold'), bg='#AAAAAA', fg='white').pack(pady=8)

        # Grade badge
        badge_frame = tk.Frame(result_card, bg='white')
        badge_frame.pack(fill='x', pady=4)
        self.grade_label = tk.Label(badge_frame, text='—',
                                    font=('Arial', 48, 'bold'),
                                    bg='white', fg='#CCCCCC')
        self.grade_label.pack()
        self.desc_label  = tk.Label(badge_frame, text='Enter values and click Predict',
                                    font=('Arial', 10, 'italic'),
                                    bg='white', fg='#888888')
        self.desc_label.pack()
        self.conf_label  = tk.Label(badge_frame, text='',
                                    font=('Arial', 10),
                                    bg='white', fg='#2E75B6')
        self.conf_label.pack()

        # Probability bars
        prob_frame = tk.Frame(result_card, bg='white')
        prob_frame.pack(fill='x', padx=20, pady=(4, 8))
        self.prob_bars: dict = {}
        for g in GRADE_ORDER:                        # FOR LOOP (Topic 3)
            row = tk.Frame(prob_frame, bg='white')
            row.pack(fill='x', pady=1)
            tk.Label(row, text=g, font=('Arial', 10, 'bold'),
                     bg='white', fg=GRADE_COLORS[g], width=3).pack(side='left')
            canvas = tk.Canvas(row, height=18, bg='#EEEEEE',
                               highlightthickness=0)
            canvas.pack(side='left', fill='x', expand=True, padx=4)
            pct_lbl = tk.Label(row, text='0.0%', font=('Arial', 9),
                                bg='white', fg='#555555', width=6)
            pct_lbl.pack(side='right')
            self.prob_bars[g] = (canvas, pct_lbl)

        tk.Label(self.root,
                 text='SMA 390 | Dr. Bernard Ondara | © 2025',
                 font=('Arial', 8, 'italic'),
                 bg='#F2F4F7', fg='#BBBBBB').pack(pady=(0, 6))

    # ── Event handlers ──
    def _on_predict(self):
        """Read inputs, validate, predict, update result panel."""
        try:
            values: list = [float(v.get()) for v in self.vars]
        except ValueError:
            messagebox.showerror('Input Error',
                                 'Please enter valid numeric values in all fields.')
            return

        try:
            result: dict = predict_grade(self.model, self.encoder, values)
        except ValueError as e:
            messagebox.showerror('Validation Error', str(e))
            return

        grade: str = result['grade']
        color: str = GRADE_COLORS[grade]

        # Update result panel (if/elif – Topic 3)
        self.result_header.configure(bg=color)
        for w in self.result_header.winfo_children():
            w.configure(bg=color)

        self.grade_label.configure(text=grade, fg=color)
        self.desc_label.configure(text=result['description'])
        self.conf_label.configure(text=f"Confidence: {result['confidence']:.1f}%")

        # Update probability bars (FOR LOOP – Topic 3)
        total_width: int = 350
        for g, (canvas, lbl) in self.prob_bars.items():
            p: float = result['probabilities'].get(g, 0.0)
            bar_w: int = max(int(p / 100 * total_width), 0)
            canvas.delete('all')
            canvas.configure(bg='#EEEEEE')
            if bar_w > 0:
                canvas.create_rectangle(0, 0, bar_w, 18,
                                         fill=GRADE_COLORS[g], outline='')
            lbl.configure(text=f'{p:.1f}%')

    def _on_reset(self):
        """Clear all input fields and reset result panel."""
        for v in self.vars:            # FOR LOOP (Topic 3)
            v.set('')
        self.result_header.configure(bg='#AAAAAA')
        for w in self.result_header.winfo_children():
            w.configure(bg='#AAAAAA')
        self.grade_label.configure(text='—', fg='#CCCCCC')
        self.desc_label.configure(text='Enter values and click Predict')
        self.conf_label.configure(text='')
        for _, (canvas, lbl) in self.prob_bars.items():
            canvas.delete('all')
            canvas.configure(bg='#EEEEEE')
            lbl.configure(text='0.0%')


# ─────────────────────────────────────────────
# ENTRY POINT  (Topic 5 – Modular structure)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('[INFO] Training model, please wait...')
    model, encoder = train_model()

    root = tk.Tk()
    app  = StudentPredictorApp(root, model, encoder)
    root.mainloop()
