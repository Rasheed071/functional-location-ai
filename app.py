# app.py
# Functional Location AI Assistant

import streamlit as st
import pandas as pd
from io import BytesIO
import os
import traceback
from typing import Dict, List, Any, Optional, Union

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.vectorstores import FAISS

# Import config values from config.py
from config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, TEMPERATURE

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Initialize Session State ---
if 'all_dfs' not in st.session_state:
    st.session_state.all_dfs = {}
if 'processed_file_names' not in st.session_state:
    st.session_state.processed_file_names = []
if 'fl_data_summary' not in st.session_state:
    st.session_state.fl_data_summary = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'error' not in st.session_state:
    st.session_state.error = None
if 'discovered_rules' not in st.session_state:
    st.session_state.discovered_rules = {} # {level: {sheet_name: [rules]}}

# --- Page Configuration and Georgia-Pacific Theming ---
GP_LOGO_PATH = "image_a88cf4.png" # Ensure this logo is in your script's directory or provide the full path

st.set_page_config(page_title="Functional Location AI Assistant", layout="wide", page_icon=GP_LOGO_PATH if os.path.exists(GP_LOGO_PATH) else "📊")

st.markdown(f"""
<style>
    /* Georgia-Pacific Inspired Theme (styles from previous response kept) */
    [data-testid="stAppViewContainer"] {{
        background: #FFFFFF; color: #333333;
    }}
    .glass-panel {{
        background: rgba(240, 242, 246, 0.8); border: 1px solid #D0D7DE;
        border-radius: 12px; padding: 25px; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }}
    .stButton > button {{
        background: #004A99 !important; color: white !important; border: none;
        padding: 0.7em 1.4em; border-radius: 6px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); transition: background-color 0.2s ease;
    }}
    .stButton > button:hover {{ background: #003A7A !important; transform: translateY(-1px); }}
    .stButton > button:active {{ background: #002D5D !important; }}
    .chat-message {{ margin-bottom: 1rem; padding: 0 10px; display: flex; }}
    .user-message-container {{ justify-content: flex-end; }}
    .ai-message-container {{ justify-content: flex-start; }}
    .message-bubble {{
        padding: 1rem; border-radius: 12px; max-width: 75%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); word-wrap: break-word;
    }}
    .user-message .message-bubble {{
        background: #D1E8FF; color: #002D72; margin-left: auto; border-bottom-right-radius: 4px;
    }}
    .ai-message .message-bubble {{
        background: #E9ECEF; color: #333333; margin-right: auto; border-bottom-left-radius: 4px;
    }}
    .message-role {{ font-weight: bold; margin-bottom: 0.3rem; color: #004A99; }}
    .user-message .message-role {{ color: #003A7A; }}
    h1, h2, h3 {{ color: #004A99; }}
    .stTabs [data-baseweb="tab-list"] button {{ color: #004A99; }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: #002D72; border-bottom-color: #002D72;
    }}
    [data-testid="stSidebar"] {{
        background-color: #F8F9FA; border-right: 1px solid #D0D7DE;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown {{
        color: #004A99;
    }}
    .rule-discovery-section h4 {{color: #003A7A; margin-top: 15px;}}
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Processing Functions (from previous response) ---
def load_and_combine_data(uploaded_files: List[Any]) -> Dict[str, pd.DataFrame]:
    dfs = {}
    processed_files_log = []
    if not uploaded_files:
        return dfs, processed_files_log
    for uploaded_file in uploaded_files:
        try:
            file_content = BytesIO(uploaded_file.getvalue())
            file_name = uploaded_file.name
            processed_files_log.append(file_name)
            if file_name.lower().endswith('.xlsx'):
                xls = pd.ExcelFile(file_content)
                for sheet_name in xls.sheet_names:
                    df = xls.parse(sheet_name)
                    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
                    unique_sheet_name = f"{file_name}_{sheet_name.strip().lower()}"
                    dfs[unique_sheet_name] = df
            elif file_name.lower().endswith('.csv'):
                df = pd.read_csv(file_content)
                df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
                unique_sheet_name = f"{file_name}_csv"
                dfs[unique_sheet_name] = df
            else:
                st.warning(f"Unsupported file type: {file_name}.")
        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
            traceback.print_exc()
    return dfs, processed_files_log

def identify_fl_columns(df_columns: List[str]) -> Dict[str, str]:
    col_map = {}
    normalized_cols = {col.lower().replace('_', ' ').replace('-', ' '): col for col in df_columns} # more normalization
    
    # Prioritize exact or common names first
    priority_map = {
        'fl': ['functional location', 'floc', 'func loc'],
        'desc': ['description', 'desc', 'text', 'floc description'],
    }
    for key, names in priority_map.items():
        for name in names:
            if name in normalized_cols:
                col_map[key] = normalized_cols[name]
                break
    
    for i in range(1, 7): # Max 6 levels considered
        level_key_norm = f'level {i}'
        level_key_alt = f'level{i}' # no space
        if level_key_norm in normalized_cols:
            col_map[f'level{i}'] = normalized_cols[level_key_norm]
        elif level_key_alt in normalized_cols: # Check for 'levelN' if 'level N' not found
             col_map[f'level{i}'] = normalized_cols[level_key_alt]

    # If standard 'functional location' not found, try to infer if a column looks like an FL ID based on level data
    if 'fl' not in col_map and col_map.get('level1'): # Basic inference
        potential_fl_col = df_columns[0] # Default to first column if no better guess
        # This inference could be much smarter based on patterns
        # For now, user should ensure their FL ID column is named clearly
        # st.sidebar.warning("Could not definitively identify main 'Functional Location' ID column. Results may vary.")


    return col_map

def process_fl_data(dfs: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if not dfs: return None
    summary = {"identified_fl_sheets": [], "hierarchical_data": [], "potential_issues": []}
    found_fl_data = False
    for sheet_name, df in dfs.items():
        fl_cols_map = identify_fl_columns(list(df.columns))
        if fl_cols_map.get('level1'): # Simplified check: if Level 1 is found, consider it for FL processing
            found_fl_data = True
            sheet_summary = {
                "sheet_name": sheet_name, "columns_found": fl_cols_map,
                "row_count": len(df), "sample_data": df.head().to_dict('records')
            }
            summary["identified_fl_sheets"].append(sheet_summary)
            for _, row in df.iterrows():
                entry = {}
                if fl_cols_map.get('fl') and pd.notna(row[fl_cols_map['fl']]): entry['functional_location'] = str(row[fl_cols_map['fl']])
                if fl_cols_map.get('desc') and pd.notna(row[fl_cols_map['desc']]): entry['description'] = str(row[fl_cols_map['desc']])
                levels_present = {f'Level {i}': str(row[fl_cols_map[f'level{i}']]) for i in range(1,7) if fl_cols_map.get(f'level{i}') and pd.notna(row[fl_cols_map[f'level{i}']])}
                if levels_present: entry['levels'] = levels_present
                if entry.get('functional_location') or entry.get('levels'): summary["hierarchical_data"].append(entry)
        else:
            summary["potential_issues"].append(f"Sheet '{sheet_name}' did not clearly map to an FL structure (e.g., missing 'Level 1' column).")
    if not found_fl_data: return None
    return summary

# --- Business Rule Discovery Function ---
def discover_business_rules_for_level(
    dfs: Dict[str, pd.DataFrame],
    target_level: int,
    fl_data_summary: Optional[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    discovered_rules_output = {}
    if not fl_data_summary or not fl_data_summary.get("identified_fl_sheets"):
        st.warning("FL data summary not available or no FL sheets identified for rule discovery.")
        return discovered_rules_output

    for sheet_info in fl_data_summary["identified_fl_sheets"]:
        sheet_name = sheet_info["sheet_name"]
        df = dfs.get(sheet_name)
        if df is None or df.empty:
            continue

        actual_fl_cols_map = sheet_info["columns_found"]
        level_col_name = actual_fl_cols_map.get(f'level{target_level}')
        next_level_col_name = actual_fl_cols_map.get(f'level{target_level + 1}')

        if not level_col_name:
            continue  # Target level column not found in this sheet's mapping

        # Filter records AT the target_level
        if next_level_col_name:
            df_at_level = df[df[level_col_name].notna() & df[next_level_col_name].isna()]
        else: # Max level for this sheet or next level not mapped
            is_max_mapped_level = True
            for i in range(target_level + 1, 7): # Max 6 levels
                if actual_fl_cols_map.get(f'level{i}'):
                    is_max_mapped_level = False
                    break
            if is_max_mapped_level:
                df_at_level = df[df[level_col_name].notna()]
            else: # Should not happen if next_level_col_name logic is sound, but as a fallback
                df_at_level = df[df[level_col_name].notna() & ~df.apply(lambda row: any(pd.notna(row[actual_fl_cols_map[f'level{i}']]) for i in range(target_level + 1, 7) if actual_fl_cols_map.get(f'level{i}')), axis=1)]


        if df_at_level.empty:
            continue

        rules_for_this_sheet = []
        
        # Columns to exclude from rule checking (level identifiers, main FL id)
        cols_defining_hierarchy = set()
        for i in range(1, 7): # Max 6 levels
            level_i_col = actual_fl_cols_map.get(f'level{i}')
            if level_i_col: cols_defining_hierarchy.add(level_i_col)
        
        main_fl_id_col = actual_fl_cols_map.get('fl')
        if main_fl_id_col: cols_defining_hierarchy.add(main_fl_id_col)
        # Optional: exclude description if it's too variable by nature
        # main_desc_col = actual_fl_cols_map.get('desc')
        # if main_desc_col: cols_defining_hierarchy.add(main_desc_col)

        potential_rule_cols = [col for col in df.columns if col not in cols_defining_hierarchy]

        for col_to_check in potential_rule_cols:
            if col_to_check in df_at_level.columns:
                # Drop NA before checking uniqueness for constant value
                # All records at this level for this column must have the same non-NA value.
                # If a column has mixed NA and one value, it's not constant across ALL records.
                # If a column is all NA, it's not a rule about a value.
                
                # We are looking for columns where ALL non-NA values are the same single value.
                # AND that this value applies to all rows in df_at_level (if not NA).
                
                # Simpler: If after dropping NA, only one unique value remains AND the count of non-NA equals len(df_at_level)
                # OR if all values are the same (including NA, if NA is the constant) - less likely what's desired.
                # User example: "all the above 6 records have the values are same" - implies no NAs for those rule fields.

                column_slice = df_at_level[col_to_check]
                if column_slice.isna().all(): # All are NaN, not a rule for a value
                    continue

                # Check if all non-NaN values are identical
                unique_values_non_na = column_slice.dropna().unique()
                if len(unique_values_non_na) == 1:
                    # Now check if these fields are non-null for ALL records at this level
                    # The user's example implies the fields forming rules are consistently populated.
                    if not column_slice.isna().any(): # No NaNs in the slice for this column
                        constant_value = unique_values_non_na[0]
                        rules_for_this_sheet.append({"field": col_to_check, "value": constant_value, "applies_to_count": len(df_at_level)})
        
        if rules_for_this_sheet:
            # Deduplicate rules by field name for this sheet
            final_rules_for_sheet = []
            seen_fields = set()
            for rule in rules_for_this_sheet:
                if rule['field'] not in seen_fields:
                    final_rules_for_sheet.append(rule)
                    seen_fields.add(rule['field'])
            if final_rules_for_sheet:
                 discovered_rules_output[sheet_name] = final_rules_for_sheet
                 
    return discovered_rules_output

# --- RAG Chain Creation ---
def create_fl_retrieval_chain(dfs_dict: Dict[str, pd.DataFrame]):
    if not OPENAI_API_KEY: st.error("OpenAI API key not set."); return None
    if not dfs_dict: st.warning("No data loaded for AI assistant."); return None
    try:
        docs = []
        for sheet_name, df in dfs_dict.items():
            # Check if this sheet was identified as an FL sheet in summary
            is_fl_sheet_in_summary = False
            if st.session_state.fl_data_summary and st.session_state.fl_data_summary.get("identified_fl_sheets"):
                if any(s_info["sheet_name"] == sheet_name for s_info in st.session_state.fl_data_summary["identified_fl_sheets"]):
                    is_fl_sheet_in_summary = True
            
            if is_fl_sheet_in_summary: # Only vectorize sheets confirmed by process_fl_data
                content = f"Content from sheet: '{sheet_name}' (File: {sheet_name.split('_')[0] if '_' in sheet_name else 'N/A'})\n\n"
                content += df.to_markdown(index=False)
                docs.append(Document(page_content=content, metadata={"source_sheet": sheet_name}))
        if not docs: st.warning("No suitable FL data found to build AI knowledge base."); return None

        embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model_name=DEFAULT_MODEL, temperature=TEMPERATURE, openai_api_key=OPENAI_API_KEY)
        template = """
        You are an expert AI assistant specialized in analyzing Functional Location (FL) data from spreadsheets.
        The provided context contains data extracted from one or more sheets, detailing FL hierarchies (e.g., Level 1 to Level 6),
        FL codes/IDs, and their descriptions, along with other attributes.

        Your tasks are to:
        1. Understand the structure and relationships within the FL data based SOLELY on the provided context.
        2. Answer user questions accurately. If information is not in the context, state that.
        3. **Business Rule Discovery Task:**
           If the user asks you to "design", "frame", "develop", "list", "find", or "discover" business rules (or constant/common/similar fields) for a specific Functional Location level (e.g., "Design business rules for Level 2 FLs" or "What fields are constant for all Level 3 entities?"):
           a. From the data in the context, identify all records that are defined *at* that specific level. A record is "at" a level (e.g., Level X) if its 'Level X' attribute is populated, AND its 'Level X+1' attribute (if such a higher level exists) is empty/not populated.
           b. For these identified records, examine all other available data fields/columns (excluding the hierarchical level identifiers like 'Level 1', 'Level 2', etc., and the main 'Functional Location' ID itself).
           c. Determine which of these examined fields have the exact same, consistently populated (not null/empty) value across ALL of the identified records for that level.
           d. Report these fields and their constant values as the discovered business rules. List each field name (column name from the data) and its corresponding constant value.
           e. If no such constant fields are found, or if the data for that level is insufficient, state that clearly.
           f. Perform this analysis only using the data present in the context. Do not invent rules or assume external knowledge.
           Example of how to report: "For Level 2 Functional Locations, the following fields were found to be constant:
           - 'Floc Category': 'M'
           - 'ABC Indicator': 'A'
           - 'Company Code': '1000'" (Use actual field names from the data).

        Context from uploaded sheet(s):
        -------------------------------
        {context}
        -------------------------------
        User Question: {question}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = ({"context": retriever, "question": RunnablePassthrough()} 
                 | prompt 
                 | llm 
                 | StrOutputParser()
                 )
        return chain
    except Exception as e: st.error(f"Error creating AI assistant: {str(e)}"); traceback.print_exc(); return None

# --- UI Components (render_chat_messages from previous response) ---
def render_chat_messages():
    for msg_idx, msg in enumerate(st.session_state.chat_history):
        container_class = 'user-message-container' if msg['role'] == 'user' else 'ai-message-container'
        bubble_class = 'user-message' if msg['role'] == 'user' else 'ai-message'
        with st.container():
            st.markdown(f"<div class='chat-message {container_class}'><div class='message-bubble {bubble_class}'>"
                        f"<div class='message-role'>{msg['role'].capitalize()}</div>"
                        f"{msg['content']}</div></div>", unsafe_allow_html=True)

def render_fl_analysis_tab():
    st.subheader("Uploaded Data Overview")
    if not st.session_state.all_dfs:
        st.info("No data loaded. Upload Excel or CSV file(s) from the sidebar."); return
    st.write(f"**Processed Files:** {', '.join(st.session_state.processed_file_names) if st.session_state.processed_file_names else 'None'}")

    if st.session_state.fl_data_summary:
        summary = st.session_state.fl_data_summary
        st.markdown("---")
        st.subheader("Functional Location Data Summary")
        if summary["identified_fl_sheets"]:
            st.write(f"Found {len(summary['identified_fl_sheets'])} sheet(s) potentially containing FL data:")
            for sheet_info in summary["identified_fl_sheets"]:
                with st.expander(f"Sheet: {sheet_info['sheet_name']} ({sheet_info['row_count']} rows)", expanded=False):
                    st.write(f"**Detected FL Columns (Normalized Key: Original Name):**")
                    st.json({k:v for k,v in sheet_info['columns_found'].items() if v}) # Show non-empty mappings
                    st.write("**Sample Data (First 5 rows):**")
                    st.dataframe(pd.DataFrame(sheet_info['sample_data']), use_container_width=True)
        # ... (hierarchical_data and potential_issues display from previous response) ...

    # --- Business Rule Discovery UI ---
    st.markdown("---")
    st.subheader("Discover Business Rules")
    st.markdown("<div class='rule-discovery-section'>", unsafe_allow_html=True)

    if not st.session_state.fl_data_summary or not st.session_state.fl_data_summary.get("identified_fl_sheets"):
        st.info("Upload and process FL data to enable rule discovery.")
    else:
        selected_level = st.selectbox(
            "Select FL Level for Rule Discovery:",
            options=list(range(1, 7)),  # Levels 1-6
            key="rule_level_selector"
        )
        if st.button(f"Discover Rules for Level {selected_level}", key=f"discover_rules_btn_lvl_{selected_level}"):
            with st.spinner(f"Discovering rules for Level {selected_level}..."):
                rules = discover_business_rules_for_level(
                    st.session_state.all_dfs,
                    selected_level,
                    st.session_state.fl_data_summary
                )
                st.session_state.discovered_rules[selected_level] = rules # Store for display
                if not rules:
                    st.info(f"No constant field business rules found for Level {selected_level} across the processed sheets, or no records exclusively at this level.")
        
        # Display discovered rules for the currently selected or previously queried level
        if selected_level in st.session_state.discovered_rules:
            rules_for_display = st.session_state.discovered_rules[selected_level]
            if rules_for_display:
                st.markdown(f"<h4>Discovered Rules for Level {selected_level}:</h4>", unsafe_allow_html=True)
                for sheet, sheet_rules in rules_for_display.items():
                    if sheet_rules:
                        st.markdown(f"**From Sheet: `{sheet}` (Applies to {sheet_rules[0]['applies_to_count']} records at this level)") # Assuming applies_to_count is same for all rules in sheet
                        rule_df_data = [{"Field Name": rule['field'], "Constant Value": rule['value']} for rule in sheet_rules]
                        st.table(pd.DataFrame(rule_df_data))
                    # else:
                    #     st.write(f"No rules found in sheet: `{sheet}` for Level {selected_level}")
            # else:
                # This case handled by the "No constant field..." message above if rules is empty after discovery
                # st.info(f"No rules were previously found or discovered for Level {selected_level}.")
        
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("---")
    st.subheader("Raw Data Sheets")
    # ... (raw data display from previous response) ...
    for name, df in st.session_state.all_dfs.items():
        with st.expander(f"View Raw Data: {name}", expanded=False):
            st.dataframe(df, use_container_width=True, height=300)

def render_assistant_tab(): # (render_assistant_tab from previous response)
    st.subheader("Chat with FL AI Assistant")
    st.write("Ask questions about the functional location data, including requests to discover business rules (e.g., 'List business rules for Level 2 FLs').")
    render_chat_messages()
    with st.form(key='chat_form', clear_on_submit=True):
        user_q = st.text_input("Your question:", placeholder="e.g., What are Level 2 locations under '1500'?", key="user_query_input")
        submitted = st.form_submit_button("Send")
        if submitted and user_q:
            if st.session_state.retrieval_chain:
                st.session_state.chat_history.append({'role': 'user', 'content': user_q})
                with st.spinner("AI is thinking..."):
                    try:
                        answer = st.session_state.retrieval_chain.invoke(user_q)
                        st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                    except Exception as e:
                        error_msg = f"Error getting AI response: {str(e)}"
                        st.error(error_msg); traceback.print_exc()
                        st.session_state.chat_history.append({'role': 'assistant', 'content': f"Sorry, an error occurred: {error_msg}"})
                st.rerun()
            else:
                st.warning("AI Assistant is not ready. Process data first.")
    if st.session_state.chat_history and st.button("Clear Chat History", key="clear_chat_main_btn"):
        st.session_state.chat_history = []; st.success("Chat history cleared!"); st.rerun()

# --- Main Application (main function from previous response, slight modifications) ---
def main():
    if os.path.exists(GP_LOGO_PATH): st.sidebar.image(GP_LOGO_PATH, width=150)
    st.sidebar.title("FL Assistant Controls")
    with st.sidebar.expander("ℹ️ About This App", expanded=False):
        st.write("""
        This AI assistant helps analyze Functional Location (FL) data.
        Upload Excel (.xlsx) or CSV (.csv) files with FL structures.
        **Features:**
        - View data summaries & raw data.
        - Discover constant field "business rules" for specific FL levels.
        - Chat with an AI to ask questions, identify patterns, and request rule discovery.
        **Expected Data:** Columns like `Level 1`-`Level 6`, `Functional Location`, `Description`, and other attributes.
        """)

    uploaded_files = st.sidebar.file_uploader("Select Excel/CSV File(s)", type=["xlsx", "csv"], accept_multiple_files=True)
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Process Data", use_container_width=True, key="proc_data_btn"):
        if uploaded_files:
            with st.spinner("Loading and processing..."):
                st.session_state.all_dfs, st.session_state.processed_file_names = load_and_combine_data(uploaded_files)
                if st.session_state.all_dfs:
                    st.success(f"Loaded: {', '.join(st.session_state.processed_file_names)}")
                    st.session_state.fl_data_summary = process_fl_data(st.session_state.all_dfs)
                    st.session_state.retrieval_chain = create_fl_retrieval_chain(st.session_state.all_dfs) # Rebuild chain
                    st.session_state.chat_history = [] # Clear chat
                    st.session_state.discovered_rules = {} # Clear previous rules
                    if st.session_state.retrieval_chain: st.success("AI Assistant is ready!")
                    else: st.error("Failed to init AI Assistant.")
                else: st.warning("No data loaded.")
            st.rerun()
        else: st.sidebar.warning("No files selected to process.")

    if col2.button("Clear All Data", use_container_width=True, key="clr_all_btn"):
        st.session_state.all_dfs = {}; st.session_state.processed_file_names = []
        st.session_state.fl_data_summary = None; st.session_state.chat_history = []
        st.session_state.retrieval_chain = None; st.session_state.discovered_rules = {}
        st.success("All data & chat cleared!"); st.rerun()

    if st.session_state.all_dfs:
         st.sidebar.info(f"📊 Analyzing: {len(st.session_state.all_dfs)} sheet(s) from {len(st.session_state.processed_file_names)} file(s).")

    st.title("🛠️ Functional Location AI Assistant")
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    if not st.session_state.all_dfs and not uploaded_files:
        col_img, col_text = st.columns([1, 3])
        with col_img:
            if os.path.exists(GP_LOGO_PATH): st.image(GP_LOGO_PATH, width=200)
            else: st.markdown("## GP", unsafe_allow_html=True)
        with col_text:
            st.markdown("### Welcome to the Functional Location AI Assistant")
            st.write("Upload FL data via the sidebar to begin analysis and chat with the AI.")
    else:
        tab1, tab2 = st.tabs(["🔍 FL Analysis & Rules", "💬 AI Assistant"])
        with tab1: render_fl_analysis_tab()
        with tab2: render_assistant_tab()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: center; margin-top: 30px; padding: 10px; color: #777;">
                   <small>FL AI Assistant &copy; 2025</small></div>""", unsafe_allow_html=True)

if __name__ == '__main__':
    if not OPENAI_API_KEY:
        st.error("🔴 Missing OPENAI_API_KEY. AI features will be disabled.")
        # Allow app to run for UI dev, but AI won't work
    main()