from flask import Flask, request, render_template_string, session
import pandas as pd
import plotly.express as px
import os, uuid, re
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure Gemini API
genai.configure(api_key="GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

# Load meteorite data
df = pd.read_csv("Meteorite_Landings.csv", encoding="ISO-8859-1", on_bad_lines="skip")
df.columns = df.columns.str.strip()
df[["reclat", "reclong"]] = df["GeoLocation"].str.extract(r"\(?\s*([-\.\d]+)[,\s]+([-\.\d]+)\s*\)?").astype(float)
df = df.dropna(subset=["name", "mass_g", "year", "reclat", "reclong"])

visual_keywords = ["bar", "line", "hist", "heat", "map", "pie", "scatter", "plot"]

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Meteorite Chat & Chart Bot</title>
    <style>
        body { font-family: Arial; background: #eef2f3; padding: 30px; }
        .wrap { background: white; padding: 20px; border-radius: 10px; max-width: 800px; margin: auto; }
        input[type=text] { width: 75%; padding: 10px; font-size: 16px; }
        input[type=submit] { padding: 10px 15px; font-size: 16px; }
        iframe { width: 100%; height: 400px; border: none; margin-top: 15px; }
        .chatbox { margin-top: 25px; text-align: left; }
        .user { font-weight: bold; color: #0077cc; }
        .bot { color: #2e7d32; margin-bottom: 10px; }
        hr { border-top: 1px solid #ccc; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="wrap">
        <h2>🌍 Meteorite Chat & Chart Bot</h2>
        <form method="POST">
            <input type="text" name="question" placeholder="Ask about meteorites or request a chart..." required>
            <input type="submit" value="Ask">
        </form>
        <div class="chatbox">
            {% for chat in history|reverse %}
                <div class="user">🧑 You: {{ chat.q }}</div>
                <div class="bot">🤖 Bot: {{ chat.a|safe }}</div>
                {% if chat.chart %}
                    <iframe src="{{ chat.chart }}"></iframe>
                {% endif %}
                <hr>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

def is_visual(q):
    return any(k in q.lower() for k in visual_keywords)

def generate_chart(q):
    filename = f"static/{uuid.uuid4().hex}.html"
    os.makedirs("static", exist_ok=True)
    q_lower = q.lower()

    if "bar" in q_lower and "mass" in q_lower:
        m = re.search(r"top\s*(\d+)", q_lower)
        n = int(m.group(1)) if m else 10
        fig = px.bar(df.nlargest(n, "mass_g"), x="name", y="mass_g", title=f"Top {n} Heaviest Meteorites")

    elif "line" in q_lower and "year" in q_lower:
        grouped = df.groupby("year")["mass_g"].sum().reset_index()
        fig = px.line(grouped, x="year", y="mass_g", title="Total Meteorite Mass by Year")

    elif "hist" in q_lower:
        fig = px.histogram(df, x="mass_g", nbins=50, title="Histogram of Meteorite Mass")

    elif "pie" in q_lower:
        pie_df = df["recclass"].value_counts().nlargest(10)
        fig = px.pie(names=pie_df.index, values=pie_df.values, title="Top 10 Meteorite Classes")

    elif "heat" in q_lower or "heatmap" in q_lower:
        heat_df = df.groupby(["reclat", "reclong"]).size().reset_index(name="count")
        fig = px.density_mapbox(heat_df, lat="reclat", lon="reclong", z="count",
                                 radius=10, center=dict(lat=0, lon=0), zoom=1,
                                 mapbox_style="stamen-terrain", title="Meteorite Heatmap")

    elif "map" in q_lower:
        fig = px.scatter_geo(df, lat="reclat", lon="reclong", hover_name="name", size="mass_g",
                             projection="natural earth", title="Meteorite Landings Around the World")

    elif "scatter" in q_lower:
        fig = px.scatter(df, x="reclong", y="reclat", color="mass_g", title="Scatter Plot of Meteorite Coordinates")

    else:
        fig = px.scatter(df, x="reclong", y="reclat", title="Default Scatter Plot")

    fig.write_html(filename)
    return filename, "✅ Here's your chart!"

def search_dataset(q):
    q = q.lower()

    # Average
    if "average mass" in q:
        avg = df["mass_g"].mean()
        return f"The average mass is {avg:,.2f} grams."

    # Heaviest meteorite
    elif "maximum mass" in q or "heaviest" in q:
        row = df.loc[df["mass_g"].idxmax()]
        return f"The heaviest meteorite is <b>{row['name']}</b>, weighing {row['mass_g']:,.2f} grams and fell in {int(row['year'])}."

    # Lightest meteorite
    elif "minimum mass" in q or "lightest" in q:
        row = df.loc[df["mass_g"].idxmin()]
        return f"The lightest meteorite is <b>{row['name']}</b>, weighing {row['mass_g']:,.2f} grams and fell in {int(row['year'])}."

    # Count of records
    elif "how many meteorites" in q or "count" in q:
        return f"There are {len(df):,} meteorite records in the dataset."

    # Meteorites found near a location (GeoLocation string match)
    elif "near" in q:
        match = re.search(r"near ([a-z\s]+)", q)
        if match:
            keyword = match.group(1).strip().lower()
            nearby = df[df["GeoLocation"].astype(str).str.lower().str.contains(keyword, na=False)]
            return nearby[["name", "mass_g", "year"]].to_html(index=False) if not nearby.empty else f"No meteorites found near {keyword.title()}."

    # Meteorites that fell in a place (matches name field)
    elif "fell in" in q or "in" in q:
        match = re.search(r"in ([a-z\s]+)", q)
        if match:
            location = match.group(1).strip().lower()
            loc_matches = df[df["name"].str.lower().str.contains(location, na=False)]
            return loc_matches[["name", "mass_g", "year"]].to_html(index=False) if not loc_matches.empty else f"No meteorites found for location: {location.title()}."

    # Year filtering
    elif "after" in q:
        match = re.search(r"after (\d{4})", q)
        if match:
            year = int(match.group(1))
            after_df = df[df["year"] > year]
            return after_df[["name", "mass_g", "year"]].to_html(index=False) if not after_df.empty else f"No meteorites found after {year}."

    elif "before" in q:
        match = re.search(r"before (\d{4})", q)
        if match:
            year = int(match.group(1))
            before_df = df[df["year"] < year]
            return before_df[["name", "mass_g", "year"]].to_html(index=False) if not before_df.empty else f"No meteorites found before {year}."

    # Top N by mass
    elif "top" in q and "mass" in q:
        m = re.search(r"top\s*(\d+)", q)
        n = int(m.group(1)) if m else 10
        top = df.nlargest(n, "mass_g")[["name", "mass_g", "year"]]
        return top.to_html(index=False)

    return None

def ask_gemini(q):
    if not model:
        return "Gemini API not configured."

    try:
        res = model.generate_content(q)
        return res.text if res and res.text else "No response generated."
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        question = request.form["question"]
        chart = ""
        if is_visual(question):
            chart, answer = generate_chart(question)
        else:
            answer = search_dataset(question)
            if not answer:
                answer = ask_gemini(question)
        session["history"].append({"q": question, "a": answer, "chart": chart})
        session.modified = True

    return render_template_string(TEMPLATE, history=session["history"])

if __name__ == "__main__":
    app.run(debug=True)
