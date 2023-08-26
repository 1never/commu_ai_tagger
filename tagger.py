
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


iso_id2tag = {"0": "Inform", "1": "Agreement", "2": "Disagreement", "3": "Correction", "4": "Answer", "5": "Confirm", "6": "Disconfirm", 
              "7": "Question", "8": "SetQuestion", "9": "PropositionalQuestion", "10": "ChoiceQuestion", "11": "CheckQuestion", 
              "12": "Offer", "13": "AddressOffer", "14": "AcceptOffer", "15": "DeclineOffer", "16": "Promise", "17": "Request", 
              "18": "AddressRequest", "19": "AcceptRequest", "20": "DeclineRequest", "21": "Suggest", "22": "AddressSuggest", 
              "23": "AcceptSuggest", "24": "DeclineSuggest", "25": "Instruct", "26": "AutoPositive", "27": "AutoNegative", 
              "28": "AlloPositive", "29": "AlloNegative", "30": "FeedbackElicitation", "31": "Stalling", "32": "Pausing", 
              "33": "InitGreeting", "34": "ReturnGreeting", "35": "InitSelfIntroduction", "36": "ReturnSelfIntroduction",
              "37": "Apology", "38": "AcceptApology", "39": "Thanking", "40": "AcceptThanking", "41": "InitGoodbye", "42": "ReturnGoodbye", "43": "Other"}

specific_id2tag = {"0": "None", "1": "SpotRequirement", "2": "SpotRelatedQuestion", "3": "SpotDetailsQuestion", "4": "RequestRecommendation", "5": "SpotImpression", "6": "CustomerExperience"}

class Tagger():
    def __init__(self, mode="iso", device="cpu"):
        self.device = device
        if mode == "specific":
            self.num_labels = 7
            modelname = "1never/line-distilbert-base-japanese-finetuned-da-specific"
            self.id2tag = specific_id2tag
        elif mode == "iso":
            self.num_labels = 44
            modelname = "1never/line-distilbert-base-japanese-finetuned-da-iso"
            self.id2tag = iso_id2tag
        else:
            raise ValueError("mode must be specific or iso")
        
        self.tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=self.num_labels)
        self.model.eval()
        self.model.to(self.device)
    
    def predict(self, context):
        if type(context) is list:
            context = "[SEP]".join(context)
        with torch.no_grad():
            out = self.model(**self.tokenizer(context, return_tensors='pt').to(self.device))
            return self.id2tag[str(torch.argmax(out["logits"]).item())]
        
# tagger = Tagger(mode="iso", device="cuda:0")
# context = ["今日はいい天気だ", "明日は雨だ"]
# print(tagger.predict(context))
