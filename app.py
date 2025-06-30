import streamlit as st
import pandas as pd
import io

# --- Backend ML Model (EventPlannerAI Class) - Integrated ---
# This class is directly included here for a self-contained Streamlit application.
# In a larger project, this would typically be in a separate Python file and imported.
class EventPlannerAI:
    """
    An AI model for recommending vendors and suggesting event planning steps,
    conceptually incorporating supervised machine learning principles for vendor scoring.
    """

    def __init__(self, vendors_csv_data):
        """
        Initializes the EventPlannerAI with vendor data.

        Args:
            vendors_csv_data (str): The content of the vendors.csv file as a string.
        """
        self.vendors_df = self._load_vendors_data(vendors_csv_data)
        self.planning_steps = self._define_planning_steps()
        # In a real supervised ML scenario, these 'weights' would be learned by a model
        # from historical data (e.g., using logistic regression, decision trees, etc.).
        # For this conceptual model, they are explicitly defined to simulate that outcome.
        self.feature_weights = {
            "event_type_match": 10,
            "budget_alignment_good": 10,
            "budget_alignment_partial": 5,
            "location_match_exact": 8,  # Increased emphasis on exact location match
            "location_match_nearby": 3, # Increased emphasis on nearby location match
            "rating_multiplier": 2 # Rating is typically 0-5, so multiply to give it similar weight scale
        }
        # Conceptual placeholder for a trained model.
        # In a real application, this would store a trained model object (e.g., from scikit-learn).
        self.trained_model = None
        # In a real supervised ML scenario, you'd call a training method here:
        # self._train_model(historical_event_data)

    def _load_vendors_data(self, csv_data):
        """
        Loads vendor data from the provided CSV string.

        Args:
            csv_data (str): The content of the vendors.csv file.

        Returns:
            pd.DataFrame: A DataFrame containing vendor information.
        """
        # Use io.StringIO to treat the string as a file
        df = pd.read_csv(io.StringIO(csv_data))
        # Clean up column names by stripping leading/trailing spaces
        df.columns = df.columns.str.strip()
        return df

    def _define_planning_steps(self):
        """
        Defines generic planning steps for various event types.

        Returns:
            dict: A dictionary mapping event types to a list of planning steps.
        """
        return {
            "Wedding": [
                "Set your wedding date and budget.",
                "Create a guest list.",
                "Choose a venue and book it.",
                "Hire a wedding planner (optional).",
                "Select caterers, photographers, decorators, and entertainment.",
                "Send out invitations.",
                "Arrange for dress/suit fittings.",
                "Plan transportation and accommodation for guests.",
                "Finalize day-of timeline."
            ],
            "Birthday": [
                "Choose a theme and date.",
                "Set a budget.",
                "Create a guest list.",
                "Select a venue or decide on home party.",
                "Order cake and food.",
                "Plan activities/entertainment.",
                "Send out invitations.",
                "Buy decorations and party favors."
            ],
            "Corporate": [
                "Define event objectives and target audience.",
                "Set a budget and date.",
                "Select a venue suitable for corporate events.",
                "Arrange catering and audio-visual equipment.",
                "Plan presentations, speakers, and networking sessions.",
                "Send out professional invitations.",
                "Manage registrations.",
                "Prepare marketing materials."
            ],
            "Exhibition": [
                "Define exhibition goals and target audience.",
                "Secure a suitable exhibition space.",
                "Develop a layout and design for your booth/area.",
                "Arrange for equipment, displays, and promotional materials.",
                "Plan staffing for the event.",
                "Market the exhibition to attract attendees.",
                "Coordinate logistics like setup and dismantle.",
                "Gather leads and feedback during the event."
            ],
            "Anniversary": [
                "Choose a significant date.",
                "Decide on the scale and intimacy of the celebration.",
                "Select a venue (restaurant, home, special location).",
                "Plan a special meal or catering.",
                "Consider sentimental decorations or activities.",
                "Arrange for photography to capture memories.",
                "Invite close family and friends."
            ],
            "baby Shower": [
                "Choose a date and time that works for the mom-to-be.",
                "Pick a theme for the baby shower.",
                "Create a guest list and send out invitations.",
                "Plan games and activities.",
                "Organize food, drinks, and cake.",
                "Decorate the venue.",
                "Prepare favors for guests."
            ],
            "Engagement": [
                "Set a date and budget.",
                "Choose a venue.",
                "Create a guest list.",
                "Select catering and entertainment.",
                "Plan an engagement ceremony or party.",
                "Arrange for rings and attire.",
                "Capture the moment with photography.",
                "Send out announcements."
            ],
            "Cultural Event": [
                "Define the purpose and theme of the cultural event.",
                "Secure a venue suitable for cultural performances or displays.",
                "Identify performers, artists, or speakers.",
                "Arrange for necessary technical support (sound, lighting).",
                "Plan for cultural food and beverage options.",
                "Market the event to the community.",
                "Coordinate with cultural organizations or groups."
            ],
            "Festival Celebration": [
                "Choose a specific festival and its dates.",
                "Determine the type of celebration (community, private).",
                "Plan traditional food and treats.",
                "Arrange for decorations and specific festival items.",
                "Organize cultural activities, music, or performances.",
                "Invite family, friends, or community members.",
                "Ensure adherence to festival customs and rituals."
            ],
            "Sports Event": [
                "Define the type of sport and competition.",
                "Secure a suitable sporting venue or field.",
                "Plan for necessary equipment and facilities.",
                "Organize teams, participants, and referees/officials.",
                "Arrange for first aid and safety measures.",
                "Market the event to attract spectators.",
                "Provide refreshments and spectator amenities.",
                "Plan for awards or recognition ceremonies."
            ],
            "Fashion Show": [
                "Define the concept and theme of the fashion show.",
                "Secure a runway and backstage area.",
                "Recruit models, designers, and stylists.",
                "Arrange for lighting, sound, and visual effects.",
                "Plan for seating and VIP areas.",
                "Market the show to attract attendees and press.",
                "Coordinate music, choreography, and show flow.",
                "Manage fittings and final preparations."
            ],
            "Music Concert": [
                "Select a date, time, and venue.",
                "Book artists or bands.",
                "Arrange for sound systems, lighting, and stage setup.",
                "Manage ticketing and promotions.",
                "Plan for security and crowd control.",
                "Provide refreshments and merchandise options.",
                "Ensure all necessary permits and licenses are acquired.",
                "Prepare a show schedule and manage backstage logistics."
            ],
            "Award ceremony": [
                "Define the categories and criteria for awards.",
                "Select a prestigious venue.",
                "Create a guest list of nominees, presenters, and attendees.",
                "Plan the ceremony flow, including speeches and entertainment.",
                "Design and order awards/trophies.",
                "Arrange for professional photography and videography.",
                "Coordinate catering and seating arrangements.",
                "Send out formal invitations."
            ],
            "College Fest": [
                "Form an organizing committee.",
                "Set a theme and budget.",
                "Secure campus venues or external locations.",
                "Plan various events (cultural, technical, sports).",
                "Invite guest speakers, performers, or judges.",
                "Arrange for sponsorships and funding.",
                "Market the fest to students and external audiences.",
                "Ensure safety and security throughout the event."
            ],
            "Invitation Designer": [
                "Determine the event type and theme.",
                "Discuss design preferences and budget with the client.",
                "Create initial design concepts and mock-ups.",
                "Incorporate client feedback and revise designs.",
                "Finalize design and select paper/printing options.",
                "Oversee production and quality control.",
                "Deliver finished invitations."
            ],
             "Stage setup": [
                "Understand the event's requirements (type, scale, audience).",
                "Design a stage layout and structure.",
                "Source necessary materials and equipment (truss, platforms, backdrops).",
                "Coordinate with lighting and sound teams.",
                "Supervise the setup and ensure safety standards.",
                "Conduct sound checks and technical rehearsals.",
                "Dismantle the stage after the event."
            ],
            "Bakery": [
                "Discuss event details and desired cake/dessert theme.",
                "Provide flavor and design options.",
                "Prepare custom cake and desserts.",
                "Arrange for delivery or pickup.",
                "Ensure food safety and presentation."
            ],
            "Florist": [
                "Consult on event theme, colors, and desired floral arrangements.",
                "Suggest seasonal and appropriate flowers.",
                "Design and create bouquets, centerpieces, and decorations.",
                "Coordinate delivery and setup at the venue.",
                "Ensure freshness and longevity of flowers."
            ],
            "Bartender": [
                "Assess event size and expected number of guests.",
                "Plan a drink menu (alcoholic and non-alcoholic).",
                "Source necessary ingredients, spirits, and bar equipment.",
                "Set up and manage the bar area.",
                "Serve drinks responsibly and efficiently.",
                "Clean up the bar area after the event."
            ],
            "Makeup Artist": [
                "Consult with the client on desired look and style.",
                "Perform skin preparation and makeup application.",
                "Ensure makeup suits the event type and client's features.",
                "Use high-quality and hygienic products.",
                "Offer touch-up tips for the event duration."
            ],
            "Photography": [
                "Discuss event schedule, key moments, and desired shots.",
                "Scout the venue for best shooting locations.",
                "Capture candid and posed photographs.",
                "Edit and retouch photos post-event.",
                "Deliver final photos in agreed-upon format and timeline."
            ],
            "Security": [
                "Assess event risks and security needs.",
                "Develop a security plan and deployment strategy.",
                "Provide trained security personnel.",
                "Manage crowd control and access points.",
                "Handle emergencies and incidents.",
                "Ensure safety of attendees, staff, and assets."
            ],
            "Catering": [
                "Discuss event type, guest count, and dietary restrictions.",
                "Propose menu options and tasting sessions.",
                "Prepare and transport food to the venue.",
                "Manage food service and presentation.",
                "Provide necessary serving staff and equipment.",
                "Clean up catering area after the event."
            ],
            "DJ": [
                "Discuss music preferences and event atmosphere.",
                "Create a playlist tailored to the event.",
                "Provide sound equipment and lighting (if needed).",
                "Engage with the audience.",
                "Manage sound levels and transitions."
            ],
            "Sound & Lights": [
                "Assess venue acoustics and lighting needs.",
                "Design sound and lighting setups.",
                "Provide and install professional audio and lighting equipment.",
                "Operate equipment during the event.",
                "Ensure optimal sound quality and visual effects."
            ],
            "Live Band": [
                "Discuss music genre and setlist preferences.",
                "Perform live music tailored to the event.",
                "Provide own instruments and sound equipment (or coordinate).",
                "Engage with the audience.",
                "Manage breaks and transitions."
            ],
            "Event Planner": [
                "Discuss overall event vision, goals, and budget.",
                "Assist with venue selection and vendor sourcing.",
                "Manage timelines, logistics, and contracts.",
                "Coordinate all event elements from planning to execution.",
                "Oversee event setup and breakdown.",
                "Handle troubleshooting and on-site management.",
                "Provide post-event follow-up."
            ],
            "Decorator": [
                "Consult on event theme, color scheme, and aesthetic.",
                "Develop design concepts and mood boards.",
                "Source and arrange decorations (linens, props, lighting, etc.).",
                "Set up and dismantle decorations at the venue.",
                "Ensure the decor aligns with the client's vision and budget."
            ],
            "Anchor": [
                "Understand the event's agenda and flow.",
                "Prepare scripts, introductions, and transitions.",
                "Engage with the audience and maintain energy.",
                "Keep the event on schedule.",
                "Handle impromptu situations professionally."
            ],
            "Mehndi Artist": [
                "Discuss desired mehndi designs and patterns.",
                "Prepare natural henna paste.",
                "Apply intricate mehndi designs on hands/feet.",
                "Provide aftercare instructions for optimal color."
            ],
            # Default planning steps for any unspecified event type
            "General": [
                "Define event purpose and goals.",
                "Set a realistic budget.",
                "Determine the date, time, and duration.",
                "Create a preliminary guest list.",
                "Research and book a suitable venue.",
                "Identify key vendors needed (catering, entertainment, etc.).",
                "Develop a marketing and communication plan.",
                "Create a detailed timeline and checklist.",
                "Plan for contingencies and unforeseen issues.",
                "Conduct post-event evaluation."
            ]
        }

    def _train_model(self, historical_event_data):
        """
        Conceptual method to simulate the training of a supervised machine learning model.
        In a real application, this method would:
        1. Preprocess 'historical_event_data' (e.g., one-hot encode categorical features, scale numerical features).
        2. Define features (X) and target variable (y, e.g., 'chosen_vendor_id' or 'vendor_suitability_score').
        3. Initialize and train a supervised learning model (e.g., scikit-learn's LogisticRegression, RandomForestClassifier, or a simple linear regression).
        4. Store the trained model in 'self.trained_model'.

        Args:
            historical_event_data (pd.DataFrame): A DataFrame containing past event details
                                                  and corresponding vendor choices/success metrics.
        """
        st.write("--- Conceptual Training Process ---")
        st.write("This method would typically train a supervised ML model using historical data.")
        st.write("1. Data preparation: Feature engineering (e.g., one-hot encoding event types, locations).")
        st.write("2. Define target: A label indicating 'best vendor' or 'vendor suitability'.")
        st.write("3. Model selection: Choose an algorithm (e.g., Logistic Regression for classification, "
              "Linear Regression for scoring, or a Decision Tree).")
        st.write("4. Training: Fit the model to the historical data to learn relationships "
              "between event parameters and successful vendor choices.")
        st.write("5. Model storage: Save the trained model for inference.")
        st.write("--- End Conceptual Training Process ---")
        # Example of how you might conceptually set weights based on 'training'
        # In a real model, these weights would be output by the training algorithm.
        # For demonstration, we keep them as pre-defined but frame them as 'learned'.
        # self.feature_weights = {
        #     "event_type_match": learned_weight_for_event_type,
        #     "budget_alignment_good": learned_weight_for_budget,
        #     ...
        # }
        self.trained_model = "Simulated Trained Model" # Placeholder


    def recommend_vendors(self, event_type, budget, location, min_rating=0):
        """
        Recommends vendors based on event type, budget, location, and minimum rating.
        The scoring here conceptually simulates a prediction from a simple linear supervised model.

        Args:
            event_type (str): The type of event (e.g., "Wedding", "Birthday").
            budget (float): The budget allocated for the event.
            location (str): The location of the event.
            min_rating (float): Minimum rating for vendors to be considered (0-5 scale).

        Returns:
            list: A list of dictionaries, each representing a recommended vendor,
                  sorted by a calculated 'suitability_score' in descending order.
        """
        if self.vendors_df.empty:
            st.error("No vendor data loaded.")
            return []

        # Convert budget to numeric, handling potential errors
        try:
            budget = float(budget)
        except ValueError:
            st.error("Invalid budget provided. Please enter a numeric value.")
            return []

        recommended_vendors = []

        for index, vendor in self.vendors_df.iterrows():
            suitability_score = 0 # This will be our predicted suitability score for the vendor

            vendor_event_types = [et.strip().lower() for et in str(vendor['Event types']).split('|')]
            input_event_type_lower = event_type.strip().lower()
            vendor_location_lower = str(vendor['Location']).strip().lower()
            input_location_lower = location.strip().lower()

            # Feature 1: Event Type Match
            if input_event_type_lower in vendor_event_types:
                suitability_score += self.feature_weights["event_type_match"]
            elif "general" in vendor_event_types or "all" in vendor_event_types:
                 suitability_score += self.feature_weights["event_type_match"] / 2 # Moderate weight

            # Feature 2: Budget Alignment
            try:
                vendor_price_range = float(vendor['Price range'])
                if budget >= vendor_price_range:
                    suitability_score += self.feature_weights["budget_alignment_good"]
                else:
                    if budget > vendor_price_range * 0.75: # If budget is at least 75% of vendor's min
                        suitability_score += self.feature_weights["budget_alignment_partial"]
            except ValueError:
                pass # Price range not a valid number, contributes 0 to budget score

            # Feature 3: Location Match
            if input_location_lower == vendor_location_lower:
                suitability_score += self.feature_weights["location_match_exact"]
            elif "mumbai" in input_location_lower and "navi mumbai" in vendor_location_lower:
                suitability_score += self.feature_weights["location_match_nearby"]

            # Feature 4: Rating Score (direct contribution to suitability)
            try:
                vendor_rating = float(vendor['Rating'])
                if vendor_rating >= min_rating:
                    suitability_score += (vendor_rating * self.feature_weights["rating_multiplier"])
                else:
                    # If rating is below min_rating, this vendor is not suitable
                    continue
            except ValueError:
                # If rating is invalid, this vendor is not suitable
                continue

            # Only add if the vendor has a positive suitability score after all checks
            if suitability_score > 0:
                recommended_vendors.append({
                    "Vendor ID": vendor['Vendor ID'],
                    "Vendor Name": vendor['Vendor Name'],
                    "Vendor Type": vendor['Vendor Type'],
                    "Location": vendor['Location'],
                    "Event types": vendor['Event types'],
                    "Price range": vendor['Price range'],
                    "Rating": vendor['Rating'],
                    "Availability": vendor['Availability'],
                    "Suitability Score": round(suitability_score, 2)
                })

        # Sort by suitability score in descending order
        recommended_vendors.sort(key=lambda x: x['Suitability Score'], reverse=True)
        return recommended_vendors

    def get_planning_steps(self, event_type):
        """
        Provides a list of suggested planning steps for a given event type.

        Args:
            event_type (str): The type of event (e.g., "Wedding", "Birthday").

        Returns:
            list: A list of strings, each representing a planning step.
        """
        return self.planning_steps.get(event_type.strip(), self.planning_steps["General"])


# --- Streamlit Frontend Application ---

st.set_page_config(page_title="Jashn!", layout="centered")

st.title("🎉 Jashnn!!")
st.markdown("Get vendor recommendations and planning steps for your next event!")

# Mock Vendor Data - Directly embedded for a self-contained Streamlit app
vendors_csv_content = """Vendor ID,Vendor Name,Vendor Type,Location,Event types,Price range,Rating,Availability
1,Haccleton,Stage setup,Indore,baby Shower,1013608,2,TRUE
2,Winyard,Bakery,Udaipur,Exhibition,3381101,4,TRUE
3,Tennock,Florist,Mumbai,Anniversary,1469581,2.7,TRUE
4,Tuther,Bartender,Chennai,baby Shower,592530,2.5,TRUE
5,Sifflett,Makeup Artist,Mumbai,Wedding,1890027,4.1,TRUE
6,Duffer,Photography,Mumbai,Exhibition,296781,2.7,TRUE
7,Hodgin,Security,Mumbai,Anniversary,3825299,2.9,TRUE
8,Abernethy,Bakery,Surat,Exhibition,510755,3.5,TRUE
9,O'Grada,Bakery,Chennai,baby Shower,529850,4.7,TRUE
10,Di Ruggiero,Invitation Designer,Surat,Anniversary,2801557,3.6,TRUE
11,Verny,Catering,Mumbai,baby Shower,3828073,2.6,TRUE
12,Barnfield,DJ,Bhopal,College Fest,551112,2.5,TRUE
13,Kivelle,Florist,Hyderbad,Engagement,1527395,4.6,TRUE
14,Tollett,Decorator,Mumbai,Exhibition,2630753,3.3,TRUE
15,Tinklin,Sound & Lights,Guwahati,Engagement,3970562,2.1,TRUE
16,Lightman,Bartender,Lucknow,Festival Celebration,3721140,1.2,TRUE
17,Vivash,Bakery,Mumbai,baby Shower,1348570,3.4,TRUE
18,Aldiss,Anchor,Jaipur,Music Concert,1888316,1.6,TRUE
19,Mayling,Makeup Artist,Udaipur,Cultural Event,3601844,1.5,TRUE
20,Macconaghy,Decorator,Mumbai,Anniversary,3857562,2.7,TRUE
21,Schurig,Florist,Navi Mumbai,baby Shower,2684498,3.3,TRUE
22,Brumfield,Decorator,Mumbai,Award ceremony,1911563,3.4,TRUE
23,Ellor,Bartender,Mumbai,Award ceremony,3224541,4.9,TRUE
24,Gearty,Mehndi Artist,Mumbai,Award ceremony,334597,2,TRUE
25,Michin,Mehndi Artist,Mumbai,Exhibition,866351,3.8,TRUE
26,Smerdon,Invitation Designer,Navi Mumbai,Award ceremony,2327442,3.7,TRUE
27,Cobon,Decorator,Navi Mumbai,Sports Event,3932438,2.8,TRUE
28,Sandes,Invitation Designer,Lucknow,Fashion Show,524269,1.4,TRUE
29,Scoullar,Decorator,Navi Mumbai,College Fest,2692049,3.7,TRUE
30,Saynor,Stage setup,Navi Mumbai,Music Concert,433657,4.6,TRUE
31,Selburn,Decorator,Mumbai,Anniversary,2431381,2.6,TRUE
32,Leavesley,Mehndi Artist,Ahmedabad,Wedding,193600,3.3,TRUE
33,Bilt,DJ,Thiruvananthapuram,Cultural Event,1733449,3.5,TRUE
34,Guerro,Invitation Designer,Mumbai,Birthday,680993,2.7,TRUE
35,Van Bruggen,Security,Navi Mumbai,Sports Event,3683454,4.6,TRUE
36,Scanderet,DJ,Mumbai,College Fest,1270648,1.1,TRUE
37,Kinnach,Anchor,Navi Mumbai,Award ceremony,486172,4.8,TRUE
38,Vawton,Invitation Designer,Kolkata,Engagement,2809306,4.8,TRUE
39,Ponder,DJ,Mumbai,Wedding,2002940,4.4,TRUE
40,Simonian,Florist,Vapi,Music Concert,1683620,1.3,TRUE
41,Kinnerley,Security,Pune,Exhibition,1902670,1.1,TRUE
42,Earie,Live Band,Navi Mumbai,Cultural Event,1173836,2,TRUE
43,Lanfare,Mehndi Artist,Surat,Fashion Show,3800624,4.4,TRUE
44,Eastcott,Florist,Kolkata,Festival Celebration,1477251,2,TRUE
45,Muggach,Security,Thiruvananthapuram,Anniversary,2934430,1.1,TRUE
46,Pau,Event Planner,Bengaluru,Wedding,2948776,1.2,TRUE
47,Tregien,Security,Guwahati,College Fest,1225211,2.4,TRUE
48,Howbrook,Stage setup,Chandigarh,Anniversary,743801,1.8,TRUE
49,Haughian,Decorator,Bhopal,Sports Event,1301824,1.8,TRUE
50,Crichten,Photography,Mumbai,baby Shower,3381129,3.5,TRUE
51,Edie,Bakery,Ranchi,Birthday,938253,4.1,TRUE
52,Schermick,Bakery,Pune,Wedding,1211534,1.1,TRUE
53,Avramov,Mehndi Artist,Lucknow,Wedding,2034778,3.2,TRUE
54,Blaszczak,Photography,Navi Mumbai,Birthday,1812348,2.2,TRUE
55,Linguard,Event Planner,Navi Mumbai,Exhibition,726692,2.7,TRUE
56,Langfitt,Catering,Mumbai,Engagement,2650533,2.7,TRUE
57,Staton,Makeup Artist,Surat,Birthday,1001195,4,TRUE
58,Claughton,Florist,Navi Mumbai,Fashion Show,3847557,3.8,TRUE
59,Woodings,Bakery,Navi Mumbai,Birthday,2015031,3.1,TRUE
60,Clawe,Bartender,Mumbai,Award ceremony,2649433,1.6,TRUE
61,Tweedy,Decorator,Mumbai,Cultural Event,1399285,1.6,TRUE
62,Le febre,Security,Mumbai,Engagement,610522,3.9,TRUE
63,Olenchikov,Invitation Designer,Ahmedabad,College Fest,1125465,2.5,TRUE
64,Brodest,Decorator,Mumbai,Exhibition,3539760,2.5,TRUE
65,Glanister,Makeup Artist,Navi Mumbai,Exhibition,2773789,1.7,TRUE
66,Whardley,Catering,Mumbai,Anniversary,3385103,3.3,TRUE
67,Cotilard,Event Planner,Mumbai,Exhibition,1701812,3.2,TRUE
68,Cowles,Catering,Mumbai,Exhibition,3871487,3.6,TRUE
69,Mattingly,Sound & Lights,Mumbai,Fashion Show,2564284,2.6,TRUE
70,Gann,Stage setup,Mumbai,Birthday,3663729,2.6,TRUE
71,Crenage,DJ,Mumbai,Engagement,2932154,1.2,TRUE
72,Pinkett,Photography,Navi Mumbai,Award ceremony,65326,4.5,TRUE
73,Castelluzzi,Anchor,Navi Mumbai,Cultural Event,1668409,4.3,TRUE
74,Sawkin,Sound & Lights,Navi Mumbai,Wedding,690641,4.4,TRUE
75,Karlmann,Photography,Mumbai,Festival Celebration,1293497,1.1,TRUE
76,Weyman,Live Band,Surat,Engagement,341333,1.7,TRUE
77,Okie,Florist,Kolkata,College Fest,1420416,3,TRUE
78,Cameli,Stage setup,Thiruvananthapuram,Anniversary,3117590,2.6,TRUE
79,Greig,Live Band,New Delhi,baby Shower,2852340,2.2,TRUE
80,Worham,Stage setup,Jaipur,baby Shower,1548665,1.4,TRUE
"""

# Initialize the AI model with the vendor data
ai_model = EventPlannerAI(vendors_csv_content)

# Define available event types and locations for dropdowns (from your CSV)
# You might want to extract these dynamically from your vendors_df if it's large
available_event_types = sorted(list(set([et.strip() for sublist in ai_model.vendors_df['Event types'].str.split('|').dropna() for et in sublist])))
# Add general types if not present
if "General" not in available_event_types:
    available_event_types.insert(0, "General") # Add general at the top
if "All" not in available_event_types:
    available_event_types.insert(1, "All") # Add All after General
available_locations = sorted(ai_model.vendors_df['Location'].unique().tolist())

# --- User Inputs ---
st.header("Tell Us About Your Event:")

col1, col2 = st.columns(2)

with col1:
    event_type_input = st.selectbox(
        "Select Event Type:",
        options=available_event_types,
        index=available_event_types.index("Wedding") if "Wedding" in available_event_types else 0,
        help="Choose the type of event you are planning."
    )

with col2:
    location_input = st.selectbox(
        "Select Event Location:",
        options=available_locations,
        index=available_locations.index("Mumbai") if "Mumbai" in available_locations else 0,
        help="Where will your event take place?"
    )

budget_input = st.number_input(
    "Enter Your Event Budget (e.g., 1000000 for 1 Million INR):",
    min_value=0,
    max_value=10000000, # Set a reasonable max value
    value=500000, # Default value
    step=100000,
    help="Enter your total budget for the event in numerical format."
)

min_rating_input = st.slider(
    "Minimum Vendor Rating (0 = Any, 5 = Excellent):",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.1,
    help="Filter vendors by a minimum average rating."
)

st.markdown("---")

if st.button("Get Recommendations & Planning Steps"):
    if not event_type_input or not location_input or budget_input is None:
        st.error("Please fill in all event details to get recommendations.")
    else:
        st.subheader(f"✨ Recommendations for {event_type_input} in {location_input}")

        # Get vendor recommendations
        with st.spinner("Finding the best vendors..."):
            recommended_vendors = ai_model.recommend_vendors(
                event_type=event_type_input,
                budget=budget_input,
                location=location_input,
                min_rating=min_rating_input
            )

        if recommended_vendors:
            st.write("Here are the top 3 vendors that best match your criteria:")
            # Display only the top 3 vendors
            top_3_vendors = recommended_vendors[:3]
            vendors_display_df = pd.DataFrame(top_3_vendors)
            # Reorder columns for display
            display_columns = [
                "Vendor Name", "Vendor Type", "Location", "Event types",
                "Price range", "Rating", "Suitability Score", "Availability"
            ]
            st.dataframe(vendors_display_df[display_columns].style.format(
                {"Price range": "{:,.0f}", "Rating": "{:.1f}", "Suitability Score": "{:.2f}"}
            ))
            st.markdown(
                """
                <style>
                .dataframe {width: 100%;}
                table.dataframe th {font-size: 14px; text-align: left;}
                table.dataframe td {font-size: 12px;}
                </style>
                """, unsafe_allow_html=True
            )
        else:
            st.warning("No vendors found matching your current criteria. Try adjusting your budget, location, or minimum rating.")

        st.subheader(f"🗓️ Planning Steps for {event_type_input}")
        # Get planning steps
        planning_steps = ai_model.get_planning_steps(event_type_input)
        if planning_steps:
            for i, step in enumerate(planning_steps):
                st.markdown(f"**{i+1}.** {step}")
        else:
            st.info("No specific planning steps available for this event type, but here are some general guidelines.")
            general_steps = ai_model.get_planning_steps("General")
            for i, step in enumerate(general_steps):
                st.markdown(f"**{i+1}.** {step}")

st.markdown("---")
st.info("This is a conceptual model. For a real-world application, the backend ML model would be trained on historical data, and the vendor data would be fetched from a persistent database.")
