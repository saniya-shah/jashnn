import streamlit as st
import pandas as pd
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Using RandomForest for advanced model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# --- Mock Vendor Data ---
# This data is embedded directly into the script for simplicity.
# In a real application, this would be loaded from a database or a more persistent source.
vendors_csv_content = '''Vendor ID,Vendor Name,Vendor Type,Location,Event types,Price range,Rating,Availability
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
'''

# --- EventPlannerAI Class (Advanced SL Model) ---
class EventPlannerAI:
    """
    An AI model for recommending vendors and suggesting event planning steps,
    now incorporating a trained RandomForestClassifier model for vendor suitability.
    """

    def __init__(self, vendors_csv_data):
        """
        Initializes the EventPlannerAI with vendor data and trains the ML model.

        Args:
            vendors_csv_data (str): The content of the vendors.csv file as a string.
        """
        self.vendors_df = self._load_vendors_data(vendors_csv_data)
        self.planning_steps = self._define_planning_steps()

        # Extract unique event types and locations from the loaded data
        self.available_event_types = sorted(list(set([et.strip() for sublist in self.vendors_df['Event types'].str.split('|').dropna() for et in sublist])))
        if "General" not in self.available_event_types:
            self.available_event_types.insert(0, "General")
        if "All" not in self.available_event_types:
            self.available_event_types.insert(1, "All")
        self.available_locations = sorted(self.vendors_df['Location'].unique().tolist())
        self.available_vendor_types = sorted(self.vendors_df['Vendor Type'].unique().tolist())

        # Train the model upon initialization
        self.trained_model_pipeline = self._train_model()

    def _load_vendors_data(self, csv_data):
        """
        Loads vendor data from the provided CSV string.

        Args:
            csv_data (str): The content of the vendors.csv file.

        Returns:
            pd.DataFrame: A DataFrame containing vendor information.
        """
        df = pd.read_csv(io.StringIO(csv_data))
        df.columns = df.columns.str.strip() # Clean up column names
        return df

    def _define_planning_steps(self):
        """
        Defines generic planning steps for various event types.

        Returns:
            dict: A dictionary mapping event types to a list of planning steps.
        """
        return {
            "Wedding": [
                "Set your wedding date and budget.", "Create a guest list.", "Choose a venue and book it.",
                "Hire a wedding planner (optional).", "Select caterers, photographers, decorators, and entertainment.",
                "Send out invitations.", "Arrange for dress/suit fittings.",
                "Plan transportation and accommodation for guests.", "Finalize day-of timeline."
            ],
            "Birthday": [
                "Choose a theme and date.", "Set a budget.", "Create a guest list.",
                "Select a venue or decide on home party.", "Order cake and food.",
                "Plan activities/entertainment.", "Send out invitations.", "Buy decorations and party favors."
            ],
            "Corporate": [
                "Define event objectives and target audience.", "Set a budget and date.",
                "Select a venue suitable for corporate events.", "Arrange catering and audio-visual equipment.",
                "Plan presentations, speakers, and networking sessions.", "Send out professional invitations.",
                "Manage registrations.", "Prepare marketing materials."
            ],
            "Exhibition": [
                "Define exhibition goals and target audience.", "Secure a suitable exhibition space.",
                "Develop a layout and design for your booth/area.",
                "Arrange for equipment, displays, and promotional materials.",
                "Plan staffing for the event.", "Market the exhibition to attract attendees.",
                "Coordinate logistics like setup and dismantle.", "Gather leads and feedback during the event."
            ],
            "Anniversary": [
                "Choose a significant date.", "Decide on the scale and intimacy of the celebration.",
                "Select a venue (restaurant, home, special location).", "Plan a special meal or catering.",
                "Consider sentimental decorations or activities.",
                "Arrange for photography to capture memories.", "Invite close family and friends."
            ],
            "baby Shower": [
                "Choose a date and time that works for the mom-to-be.", "Pick a theme for the baby shower.",
                "Create a guest list and send out invitations.", "Plan games and activities.",
                "Organize food, drinks, and cake.", "Decorate the venue.", "Prepare favors for guests."
            ],
            "Engagement": [
                "Set a date and budget.", "Choose a venue.", "Create a guest list.",
                "Select catering and entertainment.", "Plan an engagement ceremony or party.",
                "Arrange for rings and attire.", "Capture the moment with photography.", "Send out announcements."
            ],
            "Cultural Event": [
                "Define the purpose and theme of the cultural event.",
                "Secure a venue suitable for cultural performances or displays.",
                "Identify performers, artists, or speakers.",
                "Arrange for necessary technical support (sound, lighting).",
                "Plan for cultural food and beverage options.", "Market the event to the community.",
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
                "Define the type of sport and competition.", "Secure a suitable sporting venue or field.",
                "Plan for necessary equipment and facilities.",
                "Organize teams, participants, and referees/officials.",
                "Arrange for first aid and safety measures.", "Market the event to attract spectators.",
                "Provide refreshments and spectator amenities.", "Plan for awards or recognition ceremonies."
            ],
            "Fashion Show": [
                "Define the concept and theme of the fashion show.", "Secure a runway and backstage area.",
                "Recruit models, designers, and stylists.",
                "Arrange for lighting, sound, and visual effects.", "Plan for seating and VIP areas.",
                "Market the show to attract attendees and press.",
                "Coordinate music, choreography, and show flow.", "Manage fittings and final preparations."
            ],
            "Music Concert": [
                "Select a date, time, and venue.", "Book artists or bands.",
                "Arrange for sound systems, lighting, and stage setup.", "Manage ticketing and promotions.",
                "Plan for security and crowd control.", "Provide refreshments and merchandise options.",
                "Ensure all necessary permits and licenses are acquired.",
                "Prepare a show schedule and manage backstage logistics."
            ],
            "Award ceremony": [
                "Define the categories and criteria for awards.", "Select a prestigious venue.",
                "Create a guest list of nominees, presenters, and attendees.",
                "Plan the ceremony flow, including speeches and entertainment.",
                "Design and order awards/trophies.", "Arrange for professional photography and videography.",
                "Coordinate catering and seating arrangements.", "Send out formal invitations."
            ],
            "College Fest": [
                "Form an organizing committee.", "Set a theme and budget.",
                "Secure campus venues or external locations.",
                "Plan various events (cultural, technical, sports).",
                "Invite guest speakers, performers, or judges.", "Arrange for sponsorships and funding.",
                "Market the fest to students and external audiences.",
                "Ensure safety and security throughout the event."
            ],
            "Invitation Designer": [
                "Determine the event type and theme.", "Discuss design preferences and budget with the client.",
                "Create initial design concepts and mock-ups.",
                "Incorporate client feedback and revise designs.",
                "Finalize design and select paper/printing options.",
                "Oversee production and quality control.", "Deliver finished invitations."
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
                "Provide flavor and design options.", "Prepare custom cake and desserts.",
                "Arrange for delivery or pickup.", "Ensure food safety and presentation."
            ],
            "Florist": [
                "Consult on event theme, colors, and desired floral arrangements.",
                "Suggest seasonal and appropriate flowers.",
                "Design and create bouquets, centerpieces, and decorations.",
                "Coordinate delivery and setup at the venue.", "Ensure freshness and longevity of flowers."
            ],
            "Bartender": [
                "Assess event size and expected number of guests.",
                "Plan a drink menu (alcoholic and non-alcoholic).",
                "Source necessary ingredients, spirits, and bar equipment.",
                "Set up and manage the bar area.", "Serve drinks responsibly and efficiently.",
                "Clean up the bar area after the event."
            ],
            "Makeup Artist": [
                "Consult with the client on desired look and style.",
                "Perform skin preparation and makeup application.",
                "Ensure makeup suits the event type and client's features.",
                "Use high-quality and hygienic products.", "Offer touch-up tips for the event duration."
            ],
            "Photography": [
                "Discuss event schedule, key moments, and desired shots.",
                "Scout the venue for best shooting locations.", "Capture candid and posed photographs.",
                "Edit and retouch photos post-event.",
                "Deliver final photos in agreed-upon format and timeline."
            ],
            "Security": [
                "Assess event risks and security needs.", "Develop a security plan and deployment strategy.",
                "Provide trained security personnel.", "Manage crowd control and access points.",
                "Handle emergencies and incidents.", "Ensure safety of attendees, staff, and assets."
            ],
            "Catering": [
                "Discuss event type, guest count, and dietary restrictions.",
                "Propose menu options and tasting sessions.", "Prepare and transport food to the venue.",
                "Manage food service and presentation.", "Provide necessary serving staff and equipment.",
                "Clean up catering area after the event."
            ],
            "DJ": [
                "Discuss music preferences and event atmosphere.",
                "Create a playlist tailored to the event.",
                "Provide sound equipment and lighting (if needed).", "Engage with the audience.",
                "Manage sound levels and transitions."
            ],
            "Sound & Lights": [
                "Assess venue acoustics and lighting needs.", "Design sound and lighting setups.",
                "Provide and install professional audio and lighting equipment.",
                "Operate equipment during the event.", "Ensure optimal sound quality and visual effects."
            ],
            "Live Band": [
                "Discuss music genre and setlist preferences.",
                "Perform live music tailored to the event.",
                "Provide own instruments and sound equipment (or coordinate).", "Engage with the audience.",
                "Manage breaks and transitions."
            ],
            "Event Planner": [
                "Discuss overall event vision, goals, and budget.",
                "Assist with venue selection and vendor sourcing.",
                "Manage timelines, logistics, and contracts.",
                "Coordinate all event elements from planning to execution.",
                "Oversee event setup and breakdown.", "Handle troubleshooting and on-site management.",
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
                "Keep the event on schedule.", "Handle impromptu situations professionally."
            ],
            "Mehndi Artist": [
                "Discuss desired mehndi designs and patterns.",
                "Prepare natural henna paste.",
                "Apply intricate mehndi designs on hands/feet.",
                "Provide aftercare instructions for optimal color."
            ],
            "General": [
                "Define event purpose and goals.", "Set a realistic budget.",
                "Determine the date, time, and duration.", "Create a preliminary guest list.",
                "Research and book a suitable venue.",
                "Identify key vendors needed (catering, entertainment, etc.).",
                "Develop a marketing and communication plan.", "Create a detailed timeline and checklist.",
                "Plan for contingencies and unforeseen issues.", "Conduct post-event evaluation."
            ]
        }

    def _generate_mock_historical_data(self, num_events=1000): # Increased num_events for more training data
        """
        Generates synthetic historical data for training the ML model.
        Each row represents a (event_input, vendor) pair with a 'chosen_vendor' label.
        Emphasis on location and rating is increased here.
        """
        historical_records = []
        for _ in range(num_events):
            # Simulate an event's characteristics
            event_type = random.choice(self.available_event_types)
            event_budget = random.randint(50000, 4000000)
            event_location = random.choice(self.available_locations)
            min_rating_threshold = random.uniform(1.0, 4.0)

            # For each event, simulate interaction with all vendors and label them
            for _, vendor in self.vendors_df.iterrows():
                # Features for the ML model
                record = {
                    'event_type': event_type,
                    'event_budget': event_budget,
                    'event_location': event_location,
                    'vendor_type': vendor['Vendor Type'],
                    'vendor_price_range': vendor['Price range'],
                    'vendor_rating': vendor['Rating'],
                    'vendor_availability': vendor['Availability']
                }

                # Determine 'chosen_vendor' (label) based on some heuristic rules for synthetic data
                # This simulates the 'ground truth' that a supervised model would learn from
                chosen_score = 0 # Use a score to decide if chosen
                vendor_event_types = [et.strip().lower() for et in str(vendor['Event types']).split('|')]
                input_event_type_lower = event_type.strip().lower()
                vendor_location_lower = str(vendor['Location']).strip().lower()
                input_location_lower = event_location.strip().lower()

                # Rule 1: Event Type Match (strong influence)
                if input_event_type_lower in vendor_event_types:
                    chosen_score += 0.4

                # Rule 2: Budget Alignment (strong influence)
                if event_budget >= vendor['Price range']:
                    chosen_score += 0.3
                elif event_budget > vendor['Price range'] * 0.75:
                    chosen_score += 0.1

                # Rule 3: Location Match (SIGNIFICANTLY INCREASED emphasis influence)
                if input_location_lower == vendor_location_lower:
                    chosen_score += 1.0 # Very strong positive influence for exact location (increased from 0.5)
                elif "mumbai" in input_location_lower and "navi mumbai" in vendor_location_lower:
                    chosen_score += 0.5 # Moderate positive influence for nearby locations (increased from 0.2)

                # Rule 4: Rating (SIGNIFICANTLY INCREASED positive influence)
                if vendor['Rating'] >= min_rating_threshold:
                    chosen_score += (vendor['Rating'] / 5.0) * 0.7 # Scale rating contribution (increased from 0.3)

                # Rule 5: Availability (must be available)
                if not vendor['Availability']:
                    chosen_score = 0 # Cannot be chosen if not available

                # Final decision for 'chosen_vendor' label (binary classification)
                # If the combined score is high enough, consider it 'chosen'
                if chosen_score >= 1.2: # Adjusted threshold for 'chosen' due to higher weights
                    record['chosen_vendor'] = 1
                else:
                    record['chosen_vendor'] = 0

                historical_records.append(record)

        return pd.DataFrame(historical_records)


    def _train_model(self):
        """
        Trains a RandomForestClassifier model using the generated mock historical data.
        """
        st.sidebar.info("Training the ML model on synthetic data... (This happens once on app start)")
        historical_data = self._generate_mock_historical_data()

        # Define features (X) and target (y)
        X = historical_data[[
            'event_type', 'event_budget', 'event_location',
            'vendor_type', 'vendor_price_range', 'vendor_rating', 'vendor_availability'
        ]]
        y = historical_data['chosen_vendor']

        # Define preprocessing steps
        categorical_features = ['event_type', 'event_location', 'vendor_type']
        numerical_features = ['event_budget', 'vendor_price_range', 'vendor_rating']
        boolean_features = ['vendor_availability'] # Treat boolean as numerical for scaling

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features + boolean_features), # Scale boolean too
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Create a pipeline with preprocessing and RandomForestClassifier model
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
            # n_estimators: number of trees in the forest
            # class_weight='balanced': handles potential imbalance in synthetic 'chosen_vendor' labels
        ])

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model_pipeline.fit(X_train, y_train)

        # Evaluate the model (for demonstration purposes)
        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        st.sidebar.success("ML model trained successfully!")
        st.sidebar.markdown(f"**Model Evaluation (on synthetic data):**")
        st.sidebar.markdown(f"- Accuracy: `{acc:.2f}`")
        st.sidebar.markdown(f"- Precision: `{prec:.2f}`")
        st.sidebar.markdown(f"- Recall: `{rec:.2f}`")
        st.sidebar.markdown(f"- F1-Score: `{f1:.2f}`")
        st.sidebar.markdown("*(These metrics are for the synthetic training data only)*")

        return model_pipeline

    def recommend_vendors(self, event_type, budget, location, min_rating=0, top_n=3):
        """
        Recommends top N vendors based on user input, using the trained ML model.

        Args:
            event_type (str): The type of event.
            budget (float): The budget allocated.
            location (str): The event location.
            min_rating (float): Minimum rating for vendors.
            top_n (int): Number of top vendors to recommend.

        Returns:
            list: A list of dictionaries, each representing a recommended vendor,
                  sorted by suitability probability.
        """
        if self.vendors_df.empty:
            st.error("No vendor data loaded.")
            return []

        try:
            budget = float(budget)
        except ValueError:
            st.error("Invalid budget provided. Please enter a numeric value.")
            return []

        # Prepare data for prediction
        prediction_data = []
        for _, vendor in self.vendors_df.iterrows():
            # Filter by min_rating first
            try:
                vendor_rating = float(vendor['Rating'])
                if vendor_rating < min_rating:
                    continue # Skip vendors below min_rating
            except ValueError:
                continue # Skip if rating is invalid

            # Create a row for prediction, combining user input with vendor features
            prediction_data.append({
                'Vendor ID': vendor['Vendor ID'], # Include Vendor ID here
                'event_type': event_type,
                'event_budget': budget,
                'event_location': location,
                'vendor_type': vendor['Vendor Type'],
                'vendor_price_range': vendor['Price range'],
                'vendor_rating': vendor['Rating'],
                'vendor_availability': vendor['Availability']
            })

        if not prediction_data:
            return [] # No vendors left after filtering by min_rating

        prediction_df = pd.DataFrame(prediction_data)

        # Predict probabilities using the trained model
        # We want the probability of the positive class (1, i.e., 'chosen_vendor')
        suitability_probabilities = self.trained_model_pipeline.predict_proba(prediction_df.drop(columns=['Vendor ID']))[:, 1]

        # Add suitability score to the prediction_df
        prediction_df['Suitability Score'] = suitability_probabilities

        # Merge with original vendor data to get all details
        # Ensure that 'Vendor ID' is the key for merging
        recommended_vendors_df = self.vendors_df.merge(
            prediction_df[['Vendor ID', 'Suitability Score']],
            on='Vendor ID',
            how='inner' # Use inner to only keep vendors that were processed
        )

        # Sort by suitability score in descending order
        recommended_vendors_df = recommended_vendors_df.sort_values(by='Suitability Score', ascending=False)


        # Return top N vendors
        return recommended_vendors_df.head(top_n).to_dict(orient='records')

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

st.set_page_config(page_title="Event Planning AI", layout="centered")

st.title("🎉 Event Planning AI")
st.markdown("Get top vendor recommendations and planning steps for your next event!")

# Initialize the AI model with the vendor data
# Use st.experimental_singleton to cache the model instance and its loaded data
@st.experimental_singleton
def load_ai_model(csv_content):
    return EventPlannerAI(csv_content)

ai_model = load_ai_model(vendors_csv_content)

# Define available event types and locations for dropdowns (from the model)
available_event_types = ai_model.available_event_types
available_locations = ai_model.available_locations

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
        st.subheader(f"✨ Top Vendor Recommendations for {event_type_input} in {location_input}")

        # Get vendor recommendations (top 3 by default)
        with st.spinner("Finding the best vendors using the advanced ML model..."):
            recommended_vendors = ai_model.recommend_vendors(
                event_type=event_type_input,
                budget=budget_input,
                location=location_input,
                min_rating=min_rating_input,
                top_n=3 # Explicitly get top 3 as requested
            )

        if recommended_vendors:
            st.write("Here are the top 3 vendors that best match your criteria:")
            vendors_display_df = pd.DataFrame(recommended_vendors)
            display_columns = [
                "Vendor Name", "Vendor Type", "Location", "Event types",
                "Price range", "Rating", "Suitability Score", "Availability"
            ]
            st.dataframe(vendors_display_df[display_columns].style.format(
                {"Price range": "{:,.0f}", "Rating": "{:.1f}", "Suitability Score": "{:.4f}"}
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
st.info("This application uses an **advanced supervised machine learning model (Random Forest)** trained on **synthetic (mock) historical data**. In a real application, it would be trained on actual past event data and vendor success metrics.")
