BIN_SIZE = 10000
# ANALYSIS_MODE can be "all_vs_healthy" or "specific_vs_healthy"
ANALYSIS_MODE = "all_vs_healthy" 
if ANALYSIS_MODE == "specific_vs_healthy":
    SPECIFIC_GROUP = "Gastric cancer" 
    SPECIFIC_GROUP_FILE = "gastric" 
else:
    SPECIFIC_GROUP = "Pancancer"
    SPECIFIC_GROUP_FILE = "pancancer"
STRATIFY_BY = "Gender+Age" 


# Bin sizes to test: 100000, 50000, 10000, 5000