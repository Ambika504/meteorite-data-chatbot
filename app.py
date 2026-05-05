from flask import Flask, request, render_template_string, session
import pandas as pd
import plotly.express as px
import os, uuid, re
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# -----------------------------
# Gemini API (OK for local use)
# -----------------------------
genai.configure(api_key="YOUR_API_KEY")  # replace if needed
model = genai.GenerativeModel("gemini-1.5-pro")

# -----------------------------
# Load & Clean Dataset
# -----------------------------
df = pd.read_csv("Meteorite_Landings.csv", encoding="ISO-8859-1", on_bad_lines="skip")

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Auto-detect and rename mass column
for col in df.columns:
    if "mass" in col:
        df.rename(columns={col: "mass_g"}, inplace=True)

# Extract coordinates safely
df[["reclat", "reclong"]] = df["geolocation"].astype(str).str.extract(
    r"\(?\s*([-\.\d]+)[,\s]+([-\.\d]+)\s*\)?"
)

# Convert values safely
df["reclat"] = pd.to_numeric(df["reclat"], errors="coerce")
df["reclong"] = pd.to_numeric(df["reclong"], errors="coerce")
df["year"] = pd.to_datetime(df["year"], errors="coerce").dt.year

# Drop missing
df = df.dropna(subset=["name", "mass_g", "year", "reclat", "reclong"])

# -----------------------------
# UI Template
# -----------------------------
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Meteorite Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #f4f7fb;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        h2 {
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 20px;
        }

        .input-row {
            display: flex;
            gap: 10px;
        }

        input[type=text] {
            flex: 1;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 15px;
        }

        button {
            padding: 12px 18px;
            border: none;
            background: #4CAF50;
            color: white;
            border-radius: 8px;
            cursor: pointer;
        }

        button:hover {
            background: #45a049;
        }

        .examples {
            margin-top: 15px;
            font-size: 14px;
            color: #555;
        }

        .examples span {
            background: #eef3ff;
            padding: 6px 10px;
            border-radius: 6px;
            margin: 3px;
            display: inline-block;
            cursor: pointer;
        }

        .chat {
            margin-top: 25px;
        }

        .msg {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 8px;
            max-width: 80%;
        }

        .user {
            background: #d6eaff;
            margin-left: auto;
            text-align: right;
        }

        .bot {
            background: #e8f5e9;
        }

        iframe {
            width: 100%;
            height: 400px;
            border: none;
            margin-top: 10px;
            border-radius: 10px;
        }

        .loading {
            color: #888;
            font-style: italic;
        }

    </style>
</head>

<body>
<div class="container">
    <div class="card">
        <h2>🌍 Meteorite Chat & Chart Bot</h2>
        <p class="subtitle">Ask questions, analyze data, or generate charts instantly</p>

        <form method="POST" onsubmit="showLoading()">
            <div class="input-row">
                <input id="question" type="text" name="question" placeholder="Try: 'top 5 mass' or 'show map'" required>
                <button type="submit">Ask</button>
            </div>
        </form>

        <div class="examples">
            Try:
            <span onclick="fill('average mass')">average mass</span>
            <span onclick="fill('top 5 mass')">top 5 mass</span>
            <span onclick="fill('heaviest meteorite')">heaviest meteorite</span>
            <span onclick="fill('show map')">show map</span>
        </div>

        <div id="loading" class="loading" style="display:none;">Thinking...</div>

        <div class="chat">
            {% for chat in history|reverse %}
                <div class="msg user">You: {{ chat.q }}</div>
                <div class="msg bot">Bot: {{ chat.a|safe }}</div>

                {% if chat.chart %}
                    <iframe src="{{ chat.chart }}"></iframe>
                {% endif %}
            {% endfor %}
        </div>
    </div>
</div>

<script>
function fill(text) {
    document.getElementById("question").value = text;
}

function showLoading() {
    document.getElementById("loading").style.display = "block";
}
</script>

</body>
</html>
"""
# -----------------------------
# Detect Visualization
# -----------------------------
def is_visual(q):
    return any(k in q.lower() for k in ["bar", "line", "hist", "map", "pie", "scatter"])

# -----------------------------
# Generate Charts
# -----------------------------
def generate_chart(q):
    filename = f"static/{uuid.uuid4().hex}.html"
    os.makedirs("static", exist_ok=True)

    q = q.lower()

    if "bar" in q:
        fig = px.bar(df.nlargest(10, "mass_g"), x="name", y="mass_g",
                     title="Top 10 Heaviest Meteorites")

    elif "line" in q:
        grouped = df.groupby("year")["mass_g"].sum().reset_index()
        fig = px.line(grouped, x="year", y="mass_g",
                      title="Total Mass by Year")

    elif "hist" in q:
        fig = px.histogram(df, x="mass_g", nbins=50,
                           title="Mass Distribution")

    elif "map" in q:
        fig = px.scatter_geo(df, lat="reclat", lon="reclong",
                             size="mass_g", title="Meteorite Locations")

    else:
        fig = px.scatter(df, x="reclong", y="reclat")

    fig.write_html(filename)
    return filename, "Chart generated successfully!"

# -----------------------------
# Data Queries
# -----------------------------
def search_data(q):
    q = q.lower()

    # Average
    if "average mass" in q:
        return f"Average mass: {df['mass_g'].mean():,.2f} grams"

    # Heaviest
    if "heaviest" in q or "maximum" in q:
        row = df.loc[df["mass_g"].idxmax()]
        return f"Heaviest meteorite: {row['name']} ({row['mass_g']:,.2f} g)"

    # Top N
    if "top" in q and "mass" in q:
        m = re.search(r"top\s*(\d+)", q)
        n = int(m.group(1)) if m else 5
        top = df.nlargest(n, "mass_g")[["name", "mass_g", "year"]]
        return top.to_html(index=False)

    # Location-based (India, Japan, etc.)
    if "in" in q:
        match = re.search(r"in ([a-z\s]+)", q)
        if match:
            location = match.group(1).strip().lower()
            results = df[df["name"].str.lower().str.contains(location, na=False)]
            if not results.empty:
                return results[["name", "mass_g", "year"]].to_html(index=False)
            else:
                return f"No meteorites found for {location.title()}"

    # Count
    if "count" in q or "how many" in q:
        return f"Total meteorites: {len(df)}"

    return None

# -----------------------------
# Gemini AI
# -----------------------------
def ask_gemini(q):
    try:
        response = model.generate_content(q)
        return response.text if response and response.text else "No response generated."
    except Exception as e:
        return "AI not configured. Try dataset queries."

# -----------------------------
# Main Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        question = request.form["question"]

        if is_visual(question):
            chart, answer = generate_chart(question)
        else:
            answer = search_data(question)
            chart = ""
            if not answer:
                answer = ask_gemini(question)

        session["history"].append({
            "q": question,
            "a": answer,
            "chart": chart
        })
        session.modified = True

    return render_template_string(TEMPLATE, history=session["history"])

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
