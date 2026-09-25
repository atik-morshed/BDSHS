# Phase 13 Report: Qualitative Analysis

**Status**: ✅ COMPLETE

## Purpose

Select approximately 30 examples across 6 categories and document phrase recovery for hierarchical vs flat aggregation strategies.

## Categories

1. Compound insults (5 examples)
2. Multi-word insults (5 examples)
3. Sarcastic expressions (5 examples)
4. Punctuation-heavy examples (5 examples)
5. Code-mixed examples (5 examples)
6. UNK-containing examples (5 examples)

## Methodology

Since the original test dataset does not include category labels, examples were selected using:

- **Heuristic text analysis** for detectable categories (punctuation-heavy, code-mixed)
- **Random sampling** for categories requiring manual labeling (compound insults, multi-word insults, sarcastic, UNK-containing)

For each example, we report:
- Original sentence
- Ground-truth label (from test set)
- Model prediction
- Sufficiency efficiency for each strategy
- Comprehensiveness efficiency for each strategy

## Important Note

This analysis reports **phrase recovery**, not explanation correctness. Without human evaluation, we cannot claim that hierarchical explanations are "correct".

We can only observe whether hierarchical aggregation recovers contiguous phrases that flat methods treat as separate words.

## Qualitative Examples

### Compound Insults

**Number of examples**: 5

**Example ID**: 2525

**Text**: ভাই আপনি একটা মাল। খুব ভালো লাগে ভাই

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 0.1574
- Sum: 7.2145
- Mean: 7.2145
- Max: 7.2145

**Comprehensiveness Efficiency**:
- Hierarchical: 1.9435
- Sum: 7.7562
- Mean: 7.7562
- Max: 7.7562

---

**Example ID**: 2922

**Text**: এই জানুয়ারের বাচ্চার ফাসিঁ চাই

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 4.8984
- Sum: 4.8984
- Mean: 4.8473
- Max: 4.8984

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0144
- Sum: 0.0144
- Mean: 0.0810
- Max: 0.0144

---

**Example ID**: 889

**Text**: সানাই বানাই কানাই পানাই তর মুখে ইসলামের কথা মানায় না তুই সরাসরি দুধ দেখাবি আর তরে নিয়ে ওয়াজে বয়ান কর

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 5.3905
- Sum: 14.0957
- Mean: 14.2560
- Max: 14.0957

**Comprehensiveness Efficiency**:
- Hierarchical: 1.0378
- Sum: 1.6416
- Mean: -0.0012
- Max: 1.6416

---

**Example ID**: 657

**Text**: এতো সুন্দর একটা মুভির মাঝে ডিস লাইক দেয় মানুষ এর হলো আসল মাদারচোদ

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 12.5524
- Sum: 12.5524
- Mean: 12.5524
- Max: 12.5524

**Comprehensiveness Efficiency**:
- Hierarchical: -0.0802
- Sum: -0.0802
- Mean: -0.0802
- Max: -0.0802

---

**Example ID**: 3282

**Text**: জানোয়ার টাকে পেলে জুতা দিয়ে মার তাম

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 0.2277
- Sum: 6.6609
- Mean: 6.6609
- Max: 6.6609

**Comprehensiveness Efficiency**:
- Hierarchical: 0.3583
- Sum: 0.3861
- Mean: 0.3861
- Max: 0.3861

---

### Multi Word Insults

**Number of examples**: 5

**Example ID**: 838

**Text**: মাগি তুইকি ইহুদির বাচচা নগনো হয়ে মডেলিঃ করবি

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 0.0188
- Sum: 5.5925
- Mean: 5.5925
- Max: 7.0415

**Comprehensiveness Efficiency**:
- Hierarchical: 2.4474
- Sum: 0.1853
- Mean: 0.1853
- Max: 0.0078

---

**Example ID**: 2344

**Text**: এই হলো হারা মি দেশ।

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: 0.0044
- Sum: 0.0044
- Mean: 0.0044
- Max: 0.0044

**Comprehensiveness Efficiency**:
- Hierarchical: -0.0020
- Sum: -0.0020
- Mean: -0.0020
- Max: -0.0020

---

**Example ID**: 2146

**Text**: আল্লাহ সময় মত তাদের কাজের ফল দিবে। আর তা হল জাহান্নাম

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: 0.0099
- Sum: 0.0165
- Mean: 0.0165
- Max: 0.0165

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0415
- Sum: -0.0299
- Mean: -0.0299
- Max: -0.0299

---

**Example ID**: 199

**Text**: মাদারচোদের দল

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 0.0000
- Sum: 0.0379
- Mean: 0.0379
- Max: 0.0379

**Comprehensiveness Efficiency**:
- Hierarchical: 0.9778
- Sum: 1.9751
- Mean: 1.9751
- Max: 1.9751

---

**Example ID**: 2846

**Text**: আল্লাহ এই রকম ছেলের উপর গজব ফেলায়

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 5.9575
- Sum: 5.9575
- Mean: 5.9575
- Max: 5.9575

**Comprehensiveness Efficiency**:
- Hierarchical: 0.4233
- Sum: 0.4233
- Mean: 0.4233
- Max: 0.4233

---

### Sarcastic

**Number of examples**: 5

**Example ID**: 4390

**Text**: জানিনা কে কি ভাবে তোমাকে তবে আমি তোমাকে অনেক অনেক ভালো মনে করি

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: 0.0155
- Sum: 0.0501
- Mean: 0.0501
- Max: 0.0501

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0001
- Sum: 0.0004
- Mean: 0.0004
- Max: 0.0004

---

**Example ID**: 4340

**Text**: খোবোরটা ঠিক কিন্তু এক রাতের রেটকত টিজারে লিখলেন কেন আমি যদি আপনাকে বলি তোমার রেট কত তো তুমি কি বলবে 

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 10.3844
- Sum: 10.4549
- Mean: 10.3844
- Max: 10.4549

**Comprehensiveness Efficiency**:
- Hierarchical: 10.3388
- Sum: 8.8174
- Mean: 10.3388
- Max: 8.8174

---

**Example ID**: 84

**Text**: জার্রসি ওলা সালাটা ফেমাস হতে চায়ই

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 5.9190
- Sum: 5.9190
- Mean: 5.9190
- Max: 5.9190

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0123
- Sum: 0.0123
- Mean: 0.0123
- Max: 0.0123

---

**Example ID**: 1117

**Text**: দাড়িওয়ালা কাকের বাসার মত মাথা এই লোক নাকি আবার বিচারক আমার মনে হয় চ্যেনেল আইর উপর মহলের লোকেরা প্রতি

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 2.5817
- Sum: 8.6572
- Mean: 8.6459
- Max: 8.6572

**Comprehensiveness Efficiency**:
- Hierarchical: 2.3041
- Sum: 3.2907
- Mean: 2.9013
- Max: 3.2907

---

**Example ID**: 393

**Text**: জ্ঞান পাপীরা ক্ষমতায় গিয়ে সব ধ্বংস করেফেলছে। সুশিক্ষার অভাব হলে যা হয়।শালার যত্তসব

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 13.3441
- Sum: 13.3441
- Mean: 13.3441
- Max: 9.9879

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0170
- Sum: 0.0170
- Mean: 0.0170
- Max: 13.3270

---

### Punctuation Heavy

**Number of examples**: 5

**Example ID**: 549

**Text**: রুবেলরে মেয়েরা স্বামী বানানোর স্বপ্ন দেখে ????? কোন মেয়ে ভাই ???

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: -0.2332
- Sum: -0.2332
- Mean: -0.2332
- Max: -0.2332

**Comprehensiveness Efficiency**:
- Hierarchical: 0.1209
- Sum: 0.1209
- Mean: 0.1209
- Max: 0.1209

---

**Example ID**: 2687

**Text**: একতরফা স্ট্যাটাস কেউ ভালো চোখে নিবেনা।কাইটা পড়েন......

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: 0.0099
- Sum: 0.0099
- Mean: 0.0632
- Max: 0.0632

**Comprehensiveness Efficiency**:
- Hierarchical: -0.0026
- Sum: -0.0026
- Mean: -0.0019
- Max: -0.0019

---

**Example ID**: 2577

**Text**: আপনি একজন রাজাকার!!!!!! কারন রাজাকার ছাড়া এই মুক্তিযুদ্ধের সরকারের সমালোচনা কেও করতে পারে না।...... 

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 7.1868
- Sum: 9.5724
- Mean: 9.4898
- Max: 9.5724

**Comprehensiveness Efficiency**:
- Hierarchical: 7.0047
- Sum: 9.2164
- Mean: 9.4682
- Max: 9.2164

---

**Example ID**: 2653

**Text**: আজ বমি আসছে না???? নাকি বেশি করে বমির ট্যাবলেট গিলেছেন???

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: 0.0113
- Sum: 0.0113
- Mean: 0.0113
- Max: 0.0113

**Comprehensiveness Efficiency**:
- Hierarchical: -0.0006
- Sum: -0.0006
- Mean: -0.0006
- Max: -0.0006

---

**Example ID**: 1207

**Text**: ভাই আপনার কাছে কি রুবেলের,, জজ সাহেব,, পলাতক আসামী,, কাঙালী রাজা,, ধর মাস্তান,, এই ছবি গুলো আছে যদি 

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: -0.0086
- Sum: 0.0086
- Mean: 0.0079
- Max: 0.0079

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0033
- Sum: 0.0036
- Mean: 0.0627
- Max: 0.0627

---

### Code Mixed

**Number of examples**: 2

**Example ID**: 620

**Text**: আর কাওকে পেলো না হিজড়া jaky, souvik দের নিয়ে natok বানিয়েছে. Tisarনামটা দেখে প্লে করছিলাম. কিন্তু হি

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 7.1446
- Sum: 9.9606
- Mean: 9.9606
- Max: 9.6978

**Comprehensiveness Efficiency**:
- Hierarchical: 7.7841
- Sum: 10.3155
- Mean: 10.3155
- Max: 10.3122

---

**Example ID**: 487

**Text**: বিয়ের গল্প ভিদিও <a href="https://youtu.be/l59_yISJ6B0">https://youtu.be/l59_yISJ6B0</a>

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: -0.0016
- Sum: -0.0031
- Mean: 0.1176
- Max: -0.0031

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0017
- Sum: 0.0033
- Mean: -0.0023
- Max: 0.0033

---

### Unk Containing

**Number of examples**: 5

**Example ID**: 33

**Text**: একে বাংলাদেশের খেলোয়ার হিসেবে ধরা উচিত নয় কারণ সব খেলোয়াড়রা সম্মানের সাথে জড়িত আছে এখনো সময় আছ

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 1.1380
- Sum: 11.0756
- Mean: 11.0585
- Max: 11.0693

**Comprehensiveness Efficiency**:
- Hierarchical: 0.0451
- Sum: 0.0026
- Mean: 0.0001
- Max: 0.0004

---

**Example ID**: 1370

**Text**: কম হলে রাস্তায় জন পুরুষ ছিল। এ গুলা সব হিজলা। হিজলা বললে ও কম হবে।

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 0.9251
- Sum: 16.0855
- Mean: 16.0855
- Max: 16.0855

**Comprehensiveness Efficiency**:
- Hierarchical: 1.4669
- Sum: 0.5194
- Mean: 0.5194
- Max: 0.5194

---

**Example ID**: 3313

**Text**: তর মারে তুই চুদ খানকির পোলা

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 0.0255
- Sum: 0.8279
- Mean: 5.9374
- Max: 5.4521

**Comprehensiveness Efficiency**:
- Hierarchical: 1.7649
- Sum: 0.0222
- Mean: 0.0028
- Max: 0.0088

---

**Example ID**: 2796

**Text**: অাবাল চোদা

**Predicted Label**: 1

**Sufficiency Efficiency**:
- Hierarchical: 0.0000
- Sum: 1.1759
- Mean: 1.1759
- Max: 1.1407

**Comprehensiveness Efficiency**:
- Hierarchical: 0.9710
- Sum: 1.1407
- Mean: 1.1407
- Max: 1.1759

---

**Example ID**: 4099

**Text**: আল্লামুদ্দকরিয়

**Predicted Label**: 0

**Sufficiency Efficiency**:
- Hierarchical: 0.0000
- Sum: 0.0000
- Mean: 0.0000
- Max: 0.0000

**Comprehensiveness Efficiency**:
- Hierarchical: -0.1179
- Sum: -0.1179
- Mean: -0.1179
- Max: -0.1179

---

## Limitations

1. **Category labeling**: Without manual labeling, categories are approximated using heuristics or random sampling.

2. **Phrase recovery observation**: The current CSV format does not include the actual word selections or hierarchical unit structure. We report efficiency metrics but cannot observe the specific phrases recovered.

3. **Explanation correctness**: Without human evaluation, we cannot claim that hierarchical explanations are correct. We can only report that hierarchical achieves better efficiency metrics.

## Recommendations for Full Qualitative Analysis

For a complete qualitative analysis, the following steps are required:

1. **Manual category labeling**: Have domain experts label 30 examples into the 6 categories.

2. **Save word selections**: Modify the aggregation script to save which words/units were selected for each strategy.

3. **Manual evaluation**: Have human evaluators assess whether hierarchical explanations are actually more useful or correct.

4. **Phrase recovery documentation**: For each example, document the specific phrases recovered by hierarchical vs the individual words selected by flat methods.

## Conclusion

This analysis provides a framework for qualitative evaluation with 27 examples sampled across 6 categories. However, without manual labeling and word-level selection data, the analysis is limited to efficiency metrics rather than actual phrase recovery observation.

For publication-quality qualitative analysis, manual labeling and enhanced data collection are required.
