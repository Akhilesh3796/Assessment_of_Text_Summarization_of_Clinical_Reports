from flask import Flask, render_template, request
from summarization import generate_summary_gpt2, generate_summary_bart, generate_summary_T5_1,clinical_summary,rouge_scorer
from transformers import AutoTokenizer, AutoModel
from similarity import calculate_similarity_biobert, calculate_similarity_clinicalBert,perform_zero_shot_classification,roberta_relationship

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model = request.form["model"]
        similarity_model = request.form["similarity_model"]
        text_primary = request.form["text-input-primary"]
        text_secondary = request.form["text-input-secondary"]
        statement = request.form["statement"]
        text = text_primary + " " + text_secondary
        print(f"Selected model: {model}")
        print(f"Selected similarity model: {similarity_model}")
        print(f"Input text: {text_primary}")
        

        if model == "gpt2":
            summary_output,rouge = generate_summary_gpt2(text_primary,text_secondary)

        elif model == "bart":
            summary_output,rouge = generate_summary_bart(text_primary,text_secondary)
        elif model == "T5":
            summary_output,rouge = generate_summary_T5_1(text_primary,text_secondary)
        elif model == "clinicalBert":
            summary_output,rouge = clinical_summary(text_primary,text_secondary)
        else:
            print("invalid")



        print(f"Generated summary: {summary_output}")


        # Set a default value for relationship_type
        

        if similarity_model == "biobert":
            
            relationship_type  = calculate_similarity_biobert(statement, summary_output)
            

            if relationship_type < 0.6:
                relationship_type = "Contradiction"
            else:
                relationship_type = "Entailment"
    #  -------------------------------------------------------------------------------------           
        elif similarity_model == "clinicalbert":
             
            relationship_type = calculate_similarity_clinicalBert(statement, summary_output)
            
            if relationship_type < 0.6:
                relationship_type = "Contradiction"
            else:
                relationship_type = "Entailment"
    # -----------------------------------------------------------------------------------------

           
        elif similarity_model == "zero-shot":
            
            relationship_type = perform_zero_shot_classification(statement, summary_output)
            
            print(relationship_type)
            
    #--------------------------------------------------------------------------------------------- 
        elif similarity_model == "roberta":

            
            relationship_type = roberta_relationship(statement, summary_output)
            
            
            print(relationship_type)
        
        else:
            relationship_type = "unavailable"
            
        return render_template("index.html", summary_output=summary_output, text_input_primary=text_primary, text_input_secondary=text_secondary,statement=statement, relationship_type=relationship_type,roughe_score=rouge)


    return render_template("index.html")
    


if __name__ == "__main__":
    app.run(debug=True)


