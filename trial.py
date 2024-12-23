import openai
import pdfplumber
import pickle
import os
import torch
import numpy as np
import argparse
import os
import config
from transformers import AutoTokenizer, AutoModel
from model_depth import ParsingNet

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)

api_key = ""


def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text


def split_text_into_chunks(raw_text, max_chunk_size=10000):
    paragraphs = raw_text.split("\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_text_into_sections_using_gpt(raw_text):
    prompt = """I have a PDF document, and I need to split its content into logical sections based on the structure of the text. The goal is to identify sections such as headings, subheadings, paragraphs, and any numbered or bulleted lists, and extract them as separate strings while preserving all the text exactly as it appears.

    Here are the requirements:

    Each section should be extracted as a complete string.
    Maintain the exact content of the text, and it should be exhaustive.
    Use the structure of the document (e.g., headings, subheadings) to logically divide the content.
    Return the output as a dictionary where keys represent the section titles (from heading or subheading) and values are the text from those sections.
    The output should look like exactly like this (only contain the dictionary, no text before or after the dictionary, so we can run eval on it):

    {
        "Some Relevant Title": "Content of the first section...",
        "Some Other Relevant Title": "Content of the second section...",
        ...
    }
    
    Please analyze and extract the content logically but do not omit any part of the text from the PDF."""

    prompt += f"\n\nText:\n{raw_text}"

    client = openai.OpenAI(
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    response = chat_completion.choices[0].message.content
    if response.startswith("`"):
        response = response[response.index("{") :]

    if response.endswith("`"):
        response = response[: response.rindex("}") + 1]

    # print(type(eval(response)))
    return eval(response)


def process_pdf(pdf_path):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(raw_text)

    print("Total chunks: ", len(chunks))
    all_sections = []
    for i, chunk in enumerate(chunks):
        print("processing chunk ", i + 1)
        sections = split_text_into_sections_using_gpt(chunk)
        all_sections.append(sections)

    return all_sections


def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument(
        "--ModelPath",
        type=str,
        default="depth_mode/Savings/multi_all_checkpoint.torchsave",
        help="pre-trained model",
    )
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--savepath", type=str, default=base_path + "./Savings", help="Model save path"
    )
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [
        tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences
    ]
    all_segmentation_pred = []
    all_tree_parsing_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(
                input_sen_batch,
                input_EDU_breaks=None,
                LabelIndex=None,
                ParsingIndex=None,
                GenerateTree=True,
                use_pred_segmentation=True,
            )
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred


if __name__ == "__main__":

    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size
    save_path = "./prompts"

    """ BERT tokenizer and model """
    bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("xlm-roberta-base")

    bert_model = bert_model.cuda()

    for name, param in bert_model.named_parameters():
        param.requires_grad = False

    model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists("sections.pkl"):
        pdf_path = "The_Minimum_Wages_Act_1948.pdf"
        sections = process_pdf(pdf_path)

        with open("sections.pkl", "wb") as f:
            pickle.dump(sections, f)

    with open("sections.pkl", "rb") as f:
        sections = pickle.load(f)

    # Test_InputSentences = []
    # for section_dict in sections:
    #     for section in section_dict:
    #         sent = section_dict[section].replace("\n", " ")
    #         Test_InputSentences.append(sent)

    Test_InputSentences = open("./data/text_for_inference.txt").readlines()
    Test_InputSentence = " ".join(Test_InputSentences)
    Test_InputSentences = [Test_InputSentence]

    input_sentences, all_segmentation_pred, all_tree_parsing_pred = inference(
        model, bert_tokenizer, Test_InputSentences, batch_size
    )

    for index in range(len(all_segmentation_pred)):
        EDUs = []
        start = 0
        for i in range(len(all_segmentation_pred[index])):
            sent = ""
            for j in range(start, all_segmentation_pred[index][i] + 1):
                if input_sentences[index][j].startswith("▁"):
                    sent += " " + input_sentences[index][j][1:]
                else:
                    sent += input_sentences[index][j]
            start = all_segmentation_pred[index][i] + 1
            if sent.strip() == "":
                print(f"Empty sentence at index {index}, EDU {i}")
            EDUs.append(sent)

        # for k in range(len(all_tree_parsing_pred[index])):
        #     for ele in all_tree_parsing_pred[index][k].split():
        #         if ele == "NONE":
        #             continue
        #         temp = ele[1:-1].split(",")
        #         l, r = temp[0].split(":"), temp[1].split(":")
        #         if l[0] == l[-1]:
        #             if l[1].startswith("Satellite"):
        #                 EDUs[int(l[0]) - 1] = ""
        #         if r[0] == r[-1]:
        #             if r[1].startswith("Satellite"):
        #                 EDUs[int(r[0]) - 1] = ""

        EDU_STRING = ""

        for i in range(len(EDUs)):
            EDU_STRING += f"EDU {i+1}: {EDUs[i]}\n\t"

        PARSE_TREE_STRING = all_tree_parsing_pred[index]

        prompt = f"""
        I have a set of Elementary Discourse Units (EDUs) and a discourse parsing tree. Each EDU is a short text segment, and the parsing tree defines their relationships, with each EDU being labeled as either a Nucleus (key idea) or a Satellite (supporting detail).
        
        The format of the parsing tree is as follows in this example, `(1:Satellite=Contrast:4,5:Nucleus=span:6)` means the first parsing step (EDU1 to EDU6), where EDU4 is the splitting prediction, EDU1:4 (predicted as Satellite) and EDU5:6 (predicted as Nucleus) is one pair with discourse relation "Contrast"
        
        
        Our EDUs and Tree is as follows: 
        
        Here are the EDUs:

        {EDU_STRING}
        
        Here is the parsing tree:

        {PARSE_TREE_STRING}
        
        Your task:

        Use only the nucleus EDUs and do not use the satellite EDUs at all for the summary. This is to keep the summary concise and coherent. We do not need it to be exhaustive, just a coherent summary using only the nucleus EDUs.
        Write a single, coherent paragraph summarizing the topic using only the Nucleus EDUs and their type.
        Output should only have the summary, no reading between or beyond the lines, just connecting the EDUs coherently. 
        There should be nothing before or after the summary, just the summary itself.
        """
        with open(save_path + f"/trial_{index}_prompt.txt", "w") as f:
            f.write(prompt)
