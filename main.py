import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path

st.set_page_config(page_title="Project Schedule Plot", layout="wide")  # ← wider canvas
st.header("Project Schedule Plot")

# reads clipboard as table
initialization={'data':None,'read_data':False,"W_dict":dict(),"data_dict":dict(),'current_df':'','copy_df':False}
for key,value in initialization.items():
    if key not in st.session_state:
        st.session_state[key]=value
df=pd.DataFrame()
######################## Naming for df ####################################
def get_name_for_df():
    return f"Schedule_{len(st.session_state['data_dict'])}"

def clean_df(df,header_first_row=False):
    df_new = df.copy()
    if header_first_row:
        if len(df_new)>0:
            hdr = df_new.iloc[0,:].astype(str).str.strip()
            if hdr.notna().any():
                df_new.columns = hdr
                df_new = df_new.iloc[1:]
    df_new = df_new.replace("", np.nan).dropna(axis=1, how="all")
    df_new = df_new.dropna()
    return df_new
#### Upload files
uploaded = st.file_uploader("Upload .csv or .xlsx schedule files", accept_multiple_files=True, type=["csv", "xlsx", "xls"])

if st.button('Upload Files'):
    if uploaded:
        st.session_state['read_data']=True
        for up in uploaded:
            stem = Path(up.name).stem
            lower = up.name.lower()
    
            if lower.endswith(".csv"):
                try:
                    df = pd.read_csv(up)
                    key = (stem)
                    st.session_state["data_dict"][key] = clean_df(df)
                except Exception as e:
                    st.error(f"Failed to read CSV '{up.name}': {e}")
    
            else:  # Excel
                try:
                    xls = pd.ExcelFile(up)
                    for sheet in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet)
                        key = (f"{stem}__{sheet}")
                        st.session_state["data_dict"][key] = clean_df(df)
                except Exception as e:
                    st.error(f"Failed to read Excel '{up.name}': {e}")

# st.write("OR")

# if st.button("Copy Data to Streamlit Table (first row will be used as header)"):
#     st.session_state['copy_df']=True
#     df=pd.DataFrame(np.full((1, 8), "", dtype=object))
#     df_name=get_name_for_df()
#     st.session_state['data_dict'][df_name]=df
#     st.session_state['read_data']=True
    # copied_df = st.data_editor(blank_df,
    #                         use_container_width=True,
    #                         num_rows="dynamic",          # allow add/remove rows
    #                         hide_index=True,
    #                         key="editor"                 # persist edits across reruns
    #                         )

    
# Paste in Excel, then run thisdddddddddddddddd
# if st.button("Read data from clipboard"):
#     df = pd.read_clipboard()
#     if df is None or len(df)==0:
#         raise ValueError(st.error("Error: No Excel tables founded in clipboard. Please copy a table from Excel and try again"))
#     df_name=get_name_for_df()
#     st.session_state['data_dict'][df_name]=df
#     st.session_state['read_data']=True
# if st.uploader(""):
if len(st.session_state['data_dict'])>0:
    selected_df_name=st.selectbox("Select schedule",list(st.session_state["data_dict"].keys()))
    df=st.session_state['data_dict'][selected_df_name]

if len(df)>0 or st.session_state['copy_df']:
    st.subheader('Preview and Edit Data (can copy/paste new data here if needed')
    edited = st.data_editor(df,
                            use_container_width=True,
                            num_rows="dynamic",          # allow add/remove rows
                            hide_index=True,
                            key="editor"                 # persist edits across reruns
                            )
    # update dataframe
    df=clean_df(edited.copy(),False)
    
    st.session_state['data_dict'][selected_df_name]=(df)
    # st.dataframe(df)
    # try:
    start_col=[i for i in df.columns if 'start' in i.lower()][0]
    end_col=[i for i in df.columns if 'end' in i.lower()][0]
    project=[i for i in df.columns if 'whp' in i.lower() or 'proj' in i.lower()][0]

    # select column to color group
    group_col_list=[i for i in df.columns if i!=project]
    group_col=st.selectbox('Select columns used to color WHPs (CANNOT be WHP list)',group_col_list)

    df[start_col]=pd.to_datetime(df[start_col])
    df[end_col]=pd.to_datetime(df[end_col])
    df['Duration']=(df[end_col]-df[start_col]).dt.days

    ################### PlOT THE GANTT CHART ###########################
    # Color map per group
    groups = df[group_col].astype("category")
    group_names = list(groups.cat.categories)
    cmap = plt.cm.get_cmap("tab20", len(group_names))  # categorical palette
    color_map = {g: cmap(i) for i, g in enumerate(group_names)}
    if st.button("Show Plot") and st.session_state['read_data']:
        st.spinner(text="In progress...", show_time=True)
        # group_col=df.columns[-1]
        # convert to date time
        ##########
        st.subheader("Project Timeline")
        fig, ax = plt.subplots(figsize=(24, 20))
        # iterate over rows
        for i, row in df[::-1].iterrows():
            ax.barh(row[project], row["Duration"], left=row[start_col],height=0.8
            ,color=color_map[row[group_col]]
            )
        
        # --- Set ticks every 6 months ---
        locator = mdates.MonthLocator(interval=6)   # tick every 6 months
        formatter = mdates.DateFormatter("%b %Y")   # format like "Jan 2025"
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        #ax.set_title("Project Duration Timeline")
        # change axis size
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=12)
        plt.grid()
        # format x-axis as dates
        fig.autofmt_xdate()
        
        # Legend (one entry per group)
        handles = [mpatches.Patch(color=color_map[g], label=str(g)) for g in group_names]
        ax.legend(handles=handles, title="Group", loc="lower left", frameon=False,fontsize=20)
        

        st.pyplot(fig, clear_figure=True,use_container_width=True)  

        #############################################################################
        ############################## Second plot #################################
        st.subheader("Project By Year")
        # years a project is active (inclusive), then count projects per year

        years = np.arange(df[start_col].dt.year.min(), df[end_col].dt.year.max() + 1)
        if selected_df_name not in st.session_state['W_dict']:
            rows = []
            for y in years:
                y0, y1 = pd.Timestamp(y,1,1), pd.Timestamp(y+1,1,1)
                start_clip = df[start_col].clip(lower=y0, upper=y1)
                end_clip   = (df[end_col] + pd.Timedelta(days=1)).clip(lower=y0, upper=y1)  # end inclusive
                frac = (end_clip - start_clip).dt.days.clip(lower=0) / (y1 - y0).days     # per-project fraction
                rows.append(frac.to_numpy())
            
            # Year × Project fractions
            W = pd.DataFrame(np.vstack(rows), index=years, columns=df[project].astype(str))
            st.session_state['W_dict'][selected_df_name]=W

        W=st.session_state['W_dict'][selected_df_name]
        proj_to_group = df.set_index(project)[group_col]
        W_group = W.groupby(proj_to_group, axis=1).sum()
        fig2, ax2 = plt.subplots(figsize=(24, 20))
        W_group.plot(kind="bar", stacked=True, ax=ax2,figsize=(12,8),colormap='tab20',legend=True)
        ax2.set_xlabel("Year"); 
        ax2.set_ylabel("Full-year equivalents")
        
        # ax.set_title("Projects per year by group (stacked)")
        
        # ax = W.plot(kind="bar", stacked=True, figsize=(10,4))
        
        cum = W_group.cumsum(axis=1)
        bottoms = cum.shift(axis=1).fillna(0)
        
        for i, yr in enumerate(W.index):
            for col in W_group.columns:
                h = W_group.loc[yr, col]
                if h > 0:  # or a threshold like 0.05 to avoid clutter
                    y = bottoms.loc[yr, col] + h/2
                    ax2.text(i, y, f"{h:.2f}", ha="center", va="center", fontsize=12)
        
        # ax.set_xlabel("Year"); ax.set_ylabel("Full-year equivalents")
        # plt.tight_layout(); plt.show()
        
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True,use_container_width=True)  

######### Delete
    confirm = st.checkbox("Delete this Schedule")
    if st.button("🗑️ CONFIRM") and confirm:
        st.session_state["data_dict"].pop(selected_df_name, None)
        st.session_state["W_dict"].pop(selected_df_name, None)
        st.rerun()
    # except:
    #     st.write("Please copy dataframe to the grid before proceeding")
