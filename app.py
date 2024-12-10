from shiny import App, ui, reactive, render
import pandas as pd
import re
import string
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import numpy as np


CLIENT_ID = 'f5446b11104b40719d49740796e10e7a'
CLIENT_SECRET = '44092f8a55b94c558797563507ae458f'
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": "Bearer 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"}

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

podcast_data = pd.read_csv("https://raw.githubusercontent.com/Mario2747/628Group4-Spotify/refs/heads/main/podcasts.csv") 
podcast_vectors = pd.read_csv("https://raw.githubusercontent.com/Mario2747/628Group4-Spotify/refs/heads/main/podcast_metric_vectors.csv") 
categories = podcast_data['Category'].unique() 

# Hugging Face API 
def hf_query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"labels": ["unknown"], "scores": [0.0]}


def clean_description(description):
    if not isinstance(description, str):
        return ""
    description = description.lower()
    description = re.sub(r"http\S+", "", description) 
    description = re.sub(f"[{string.punctuation}]", " ", description)  
    return description

def search_podcasts(search_term, limit=10):
    results = sp.search(q=search_term, type='episode', limit=limit)
    episodes = results.get('episodes', {}).get('items', [])
    
    if not episodes:
        return []
    
    podcast_data = []
    for episode in episodes:
        podcast_data.append({
            'name': episode.get('name', 'N/A'),
            'description': episode.get('description', ''),
            'release_date': episode.get('release_date', 'N/A'),
            'url': episode.get('external_urls', {}).get('spotify', 'N/A'),
            'image': episode.get('images', [{}])[0].get('url', '') 
        })
    
    return podcast_data

def classify_podcasts(podcast_data):

    # change to list
    if isinstance(podcast_data, dict):
        podcast_data = [podcast_data]

    if not podcast_data:
        return pd.DataFrame(columns=['name', 'description', 'category', 'confidence', 'release_date', 'url', 'image', 'scores'])

    candidate_labels = ["science", "entertainment", "education", "health", 
                        "sports", "culture", "politics", "business", "comedy", "art"]
    
    data = []
    for podcast in podcast_data:
        description = podcast.get('description', '')
        cleaned_description = clean_description(description) 
        image_url = podcast.get('images', [{}])[0].get('url', '')
        if cleaned_description:

            classification = hf_query({
                "inputs": cleaned_description,
                "parameters": {"candidate_labels": candidate_labels},
            })
            if classification:
                max_index = classification['scores'].index(max(classification['scores']))
                category = classification['labels'][max_index]
                confidence = classification['scores'][max_index]
            else:
                category = "unknown"
                confidence = 0.0
        else:
            category = "unknown"
            confidence = 0.0
        
        data.append({
            'name': podcast.get('name', 'N/A'),
            'description': description,
            'category': category,
            'confidence': confidence,
            'release_date': podcast.get('release_date', 'N/A'),
            'url': podcast.get('external_urls', {}).get('spotify', 'N/A'),
            'image': image_url,
            'scores': classification['scores'] if cleaned_description else [0] * len(candidate_labels)
        })
    
    return pd.DataFrame(data)



# radar
def plot_interactive_radar(scores, labels, podcast_name):
    scores = scores + scores[:1] 
    labels = labels + labels[:1]  
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=labels,
        fill='toself',
        name=podcast_name,
        line=dict(color="blue")
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        width=400,
        height=400
    )
    return pio.to_html(fig, full_html=False)

def calculate_euclidean_distance(target_vector, vectors):

    return np.linalg.norm(vectors - target_vector, axis=1)

def find_closest_podcasts(target_scores, top_n=20):

    category_columns = ["Science", "Entertainment", "Education", "Health", "Sports", 
                        "Culture", "Politics", "Business", "Comedy", "Art"]
    vectors = podcast_vectors[category_columns].values
    distances = calculate_euclidean_distance(np.array(target_scores), vectors)
    podcast_vectors['distance'] = distances
    closest_podcasts = podcast_vectors.nsmallest(top_n, 'distance')
    return closest_podcasts

app_ui = ui.page_fluid(
    
    ui.tags.style("""
        body {
            font-family: Josefin Sans;
            font-weight: bold; /* 字体加粗 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 字体阴影 */
            background-image: url('https://static.vecteezy.com/system/resources/previews/047/918/672/non_2x/music-abstract-with-headphones-horizontal-wallpaper-photo.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: black;
        }
        .classification-container {
            display: grid; 
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            justify-content: center; 
            min-width: 1000px;
            width: 90vh;
            margin: 0 auto;
            padding: 20px;
            min-height: 800px; 
            height: 80vh;
            box-sizing: border-box;
        }
        .classification-row {
            display: flex;
            align-items: center;
            width: 100%; 
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.65); 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .selected {
            background-color: #4CAF50; 
            color: white;  
            border-color: #4CAF50;  
            margin-top: 20px;
        }     

        .podcasts-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            width: 1000px;
            max-width: 100%; 
            border: 1px solid #ccc; 
            border-radius: 10px; 
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.65);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
        }


        .podcast-row {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%; 
        }


        .podcast-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%; 
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd; 
        }

        .podcast-image {
            width: 100px; 
            height: 100px;
            object-fit: cover;
            margin-right: 20px;
            border-radius: 10px;
            border: 1px solid #ddd; 
        }

        .podcast-info {
            flex-grow: 1;
            text-align: left;
            font-size: 14px; 
        }

        .charts-container {
            display: flex;
            justify-content: space-between; 
            align-items: flex-start;
            width: 100%;
            margin-top: 20px;
            gap: 10px; 
        }

        .radar-chart-container,
        .scatter-chart-container {
            width: 48%;
            max-width: 48%; 
            height: 300px; 
            padding: 10px; 
            background-color: #f9f9f9; 
            border: 1px solid #ddd; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            overflow: hidden; 
            display: flex;
            align-items: center; 
            justify-content: center; 
        }
        .sidebar {
            background-color: rgba(255, 255, 255, 1);
            padding: 20px; 
            border: 1px solid #ccc; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
            width: 300px; 
            max-width: 90%; 
            margin-right: 20px; 
        }
        /* nav bar */
        .nav-tabs-container .nav-tabs {
            color: white; 
        }

        .nav-tabs-container .nav-tabs a {
            color: white; 
            text-decoration: none; 
        }

        .nav-tabs-container .nav-tabs a:hover {
            color: #315491;
        }

        .nav-tabs-container .nav-tabs .active a {
            font-weight: bold; 
            color: white; 
        }

        /* category */
        .category-button {
            background-color: rgba(240, 165, 25, 0.8); 
            color: white;
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px;
            font-size: 14px;
            cursor: pointer; 
            transition: background-color 0.3s ease; 
            margin-top: 20px;
        }

        .category-button:hover {
            background-color: darkgray; 
        }

        .category-button.selected {
            background-color: black; 
            color: white; 
        }


        @keyframes fadeIn {
            from {
                opacity: 0; 
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0); 
            }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-in-out;
        }
                  
        .footer {
            position: relative; 
            width: 100%;
            padding: 10px 0;
            color: white;
            text-align: center;
            margin-top: 20px; 
        }

        .footer-content p {
            margin: 2px 0; 
            font-size: 10px; 
        }


                  
    """),


    ui.div(
        ui.navset_tab(
            ui.nav_panel("Home", ui.page_fluid(
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h3("Podcast Search"),
                        ui.input_text("keyword", "Keyword:", ""),
                        ui.input_numeric("limit", "number:", 5, min=1),
                        ui.input_action_button("submit", "Search"),
                        ui.output_ui("dropdown_ui"),
                        style="background-color: rgba(255, 255, 255, 0.65)"
                    ),
                    ui.output_ui("show_result")  
                )
            )),
            ui.nav_panel("Category", ui.page_fluid(
                ui.div(
                    ui.output_ui("category_buttons"),
                    ui.div(
                        ui.output_ui("category_podcasts_ui"),
                        ui.output_ui("closest_podcasts_chart"),
                        class_="results-container"
                    ),
                    class_="results-container fade-in"
                )
            ))
        ),
        class_="nav-tabs-container fade-in"  # 包装父级元素，添加类
    ),
    ui.div(
        ui.div(
            ui.p("Email: rma235@wisc.edu"),
            ui.p("Designed by Mario Ma, Yiteng Tu"),
            class_="footer-content"
        ),
        class_="footer"
    )

)

def server(input, output, session):
    reactive.error_message = reactive.Value("")
    reactive.closest_podcasts = reactive.Value([])
    reactive.selected_category = reactive.Value("entertainment")

    @reactive.Value
    def search_results():
        return []

    @reactive.Effect
    @reactive.event(input.submit)
    def perform_search():
        try:
            if not input.keyword():
                raise ValueError("Please input keyword.")
            
            results = search_podcasts(input.keyword(), int(input.limit()))
            if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
                raise ValueError("There is no result.")
            
            search_results.set(results)
        except Exception as e:
            search_results.set([])
            reactive.error_message.set(f"Searching failed: {str(e)}")


    @output
    @render.ui
    def dropdown_ui():
        results = search_results()
        if not results or not isinstance(results, list):
            if reactive.error_message():
                return ui.p(reactive.error_message())
            return ui.p("Please choose your keyword.")
        
        options = {index: result["name"] for index, result in enumerate(results) if "name" in result}
        if not options:
            return ui.p("Searching failed.")
        
        return ui.input_select("dropdown_selected_result", "Choose your podcast:", options)

    @output
    @render.ui
    def show_result():
        if not input.dropdown_selected_result():
            return ui.p("Please choose your podcast.")
        
        try:
            results_new = search_results()

            if not results_new:
                return ui.p("Searching failed.")

            selected_index = int(input.dropdown_selected_result())
            if selected_index < 0 or selected_index >= len(results_new):
                return ui.p("Index is out of range")
            
            selected_row = results_new[selected_index]

            row = classify_podcasts(selected_row)
            if row.empty:
                return ui.p("Searching failed.")

            target_scores = row.iloc[0].get('scores', [])
            if not target_scores:
                return ui.p("Information is not enough to visualization, please change a podcast.")

            closest_podcasts = find_closest_podcasts(target_scores, top_n=20)
            reactive.closest_podcasts.set(closest_podcasts)

            scatter_fig = px.scatter(
                closest_podcasts,
                x="distance",
                y="Podcast_Name",
                size="distance", 
                color="distance"
            )

            scatter_fig.update_traces(
                marker=dict(sizeref=0.5, sizemin=5) 
            )

            scatter_fig.update_layout(
                width=400, 
                height=300,  
                margin=dict(l=10, r=10, t=30, b=10), 
                font=dict(
                    family="Arial, sans-serif",  
                    size=10, 
                    color="black"  
                ),
                xaxis=dict(
                    title=dict(font=dict(size=10)), 
                    tickfont=dict(size=9)
                ),
                yaxis=dict(
                    title=dict(font=dict(size=10)),
                    tickfont=dict(size=9) 
                ),
                legend=dict(
                    font=dict(size=9), 
                    title_font=dict(size=10) 
                )
            )

            scatter_html = scatter_fig.to_html(full_html=False)


            podcast_row = ui.div(

                ui.div(
                    ui.img(src=selected_row['image'], class_="podcast-image"),
                    ui.div(
                        ui.h4(f"{selected_row['name']}"),
                        ui.p(f"Categoty: {row.iloc[0]['category']}"),
                        ui.p(f"Confidence: {row.iloc[0]['confidence']:.2f}"),
                        ui.p(f"Release date: {selected_row['release_date']}"),
                        class_="podcast-info"
                    ),
                    class_="podcast-header"
                ),

                ui.div(
                    ui.div(
                        ui.HTML(plot_interactive_radar(
                            row.iloc[0]['scores'],
                            ["science", "entertainment", "education", "health", "sports", "culture", "politics", "business", "comedy", "art"],
                            row.iloc[0]['name']
                        )),
                        class_="radar-chart-container"
                    ),
                    ui.div(
                        ui.HTML(scatter_html),
                        class_="scatter-chart-container"
                    ),
                    class_="charts-container"
                ),
                class_="podcast-row"
            )

            return ui.div(podcast_row, class_="podcasts-container fade-in")
        except Exception as e:
            return ui.p(f"Going wrong: {str(e)}")
    
    @output
    @render.ui
    def category_buttons():
        buttons = []
        for category in categories:
            sanitized_category = category.replace(" ", "_")  
            is_selected = reactive.selected_category.get() == category
            button_class = "selected" if is_selected else "category-button" 

            buttons.append(
                ui.input_action_button(
                    f"btn_{sanitized_category}", category, class_=button_class
                )
            )
        
        return ui.div(*buttons, style="display: flex; flex-wrap: wrap; gap: 10px;")


    for category in categories:
        sanitized_category = category.replace(" ", "_")
        
        @reactive.Effect
        def update_selected_category(category=category, sanitized_category=sanitized_category):
            if input[f"btn_{sanitized_category}"]():
                reactive.selected_category.set(category)

    @output
    @render.ui
    def category_podcasts_ui():
        selected_category = reactive.selected_category.get()
        if selected_category is None:
            return ui.div("Please choose a category.", class_="classification-container")

        filtered_data = podcast_data[podcast_data['Category'] == selected_category]

        podcast_cards = []
        for _, row in filtered_data.iterrows():
            card = ui.div(
                ui.div(
                    ui.img(
                        src=row["Cover_URL"],
                        style="width: 100px; height: 100px; object-fit: cover; border-radius: 10px;"
                    ),
                    style="flex-shrink: 0; margin-right: 10px;"
                ),
                ui.div(
                    ui.h5(row["Podcast_Name"]),
                    ui.p(f"Publisher: {row['Publisher']}"),
                    ui.p(f"Episodes: {row['Episode_Count']}"),
                ),
                class_="classification-row fade-in"
            )
            podcast_cards.append(card)


        return ui.div(
            *podcast_cards,
            class_="classification-container fade-in"
        )



app = App(app_ui, server)
