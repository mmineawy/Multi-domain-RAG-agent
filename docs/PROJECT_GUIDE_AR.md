# دليل المشروع (للاستخدام الشخصي)

## 1) المشروع ده إيه؟
Agent واحد بيجاوب من مستندات بتديهاله (RAG)، ومش من دماغه. بتغيّر المجال (Medical / Pharma / Marketing) من غير ما تغيّر الكود، بس بملف YAML وفولدر مستندات.

## 2) الـ Pipeline خطوة خطوة
1. المستخدم يكتب سؤال في الشات.
2. **Input Guardrail:** لو السؤال تشخيص أو جرعة شخصية أو خداع أو prompt injection يترفض. ولو طوارئ (ألم صدر مثلاً) يطلع رسالة طوارئ.
3. **Retrieval:** السؤال يتحوّل لأرقام (embedding)، ونجيب أقرب 4 مقاطع من المستندات من ChromaDB.
4. **Grounding Gate:** لو أعلى تشابه أقل من 0.58 يرد "مفيش معلومات كفاية" **من غير ما يكلّم الـ LLM** (أسرع وبيوفّر quota).
5. **LLM (Gemini):** بياخد الـ system prompt + few-shot + المقاطع + السؤال، ويرد JSON منظم.
6. **Output Validation:** نتأكد من شكل الـ JSON، وبنشيل أي مصدر الموديل اخترعه، ونبني الـ citations (اسم الملف + رقم الصفحة) من اللي اترجع فعلاً.
7. **Second Gate:** لو الموديل نفسه مش واثق (أقل من 0.3) يبقى "مفيش معلومات".
8. **Confidence:** 60% قوة الـ retrieval + 40% تقدير الموديل، ويطلع High أو Medium أو Low.
9. الرد النهائي: ملخص + نقاط + توصيات + مخاطر + مصادر بصفحاتها + confidence + disclaimer.

## 3) كل ملف بيعمل إيه

| الملف | الدور |
|---|---|
| `app.py` | واجهة Streamlit: اختيار الدومين والـ Task، عرض الرد والمصادر والـ confidence |
| `cli.py` | نفس الشات في التيرمينال |
| `ingest.py` | يبني الـ index من المستندات. تدريجي: يعمل embedding للجديد بس، ويكمّل لو وقف |
| `check_setup.py` | فحص صحة المشروع (البيئة، المفتاح، الموديلات، الـ index، الـ guardrails) |
| `calibrate.py` | يحسب أنسب قيمة لـ `MIN_RELEVANCE` من الـ golden set |
| `quota_test.py` | يشخّص مشاكل الـ 429 والـ quota |
| `dagent/agent.py` | قلب المشروع: الـ pipeline كله |
| `dagent/rag.py` | قراءة PDF/CSV/TXT/MD، التقطيع (chunking)، التخزين في Chroma، البحث |
| `dagent/guardrails.py` | فلترة المدخلات، حساب الـ confidence، التحقق من الرد |
| `dagent/llm.py` | الاتصال بـ Gemini (chat + embeddings)، وإعادة المحاولة، والموديلات الاحتياطية |
| `dagent/config.py` | قراءة الإعدادات من `.env` |
| `dagent/tools.py` | الـ Calculator: حساب دقيق (BMI وMAP وROI وCAC وNNT وحساب عام) |
| `dagent/memory.py` | ذاكرة المستخدم: دوره ومستوى التفصيل واللغة وآخر مواضيعه، محفوظة في `user_profiles/` |
| `evaluation/question_forge.py` | بيولّد أسئلة اختبار من مستنداتك (تراجعها بإيدك) |
| `domains/*.yaml` | الدومين كله: system prompt، few-shot، الـ tasks، الأنماط الممنوعة، الـ disclaimer |
| `knowledge_base/<domain>/` | مستندات كل دومين |
| `knowledge_base_samples/` | عينات تجريبية خيالية |
| `evaluation/golden_set.json` | أسئلة الاختبار (عادية، ممنوعة، برة المجال) |
| `evaluation/run_eval.py` | يشغّل الأسئلة ويطلّع جدول نجاح/فشل |
| `docs/` | الـ architecture والـ design document ودليلك ده |

## 4) الأوامر اللي هتستخدمها
```
.venv\Scripts\activate
python check_setup.py                      # فحص
python ingest.py medical                   # بناء/تحديث index (دومين واحد)
streamlit run app.py                       # الواجهة
python -m evaluation.run_eval --domain medical
python calibrate.py                        # بعد ما تحط المستندات الحقيقية
```

## 5) احنا عملنا إيه ومعملناش إيه

**✅ خلص**
- LLM (Gemini) + RAG + Prompt engineering (system prompt وfew-shot لكل دومين) + Guardrails.
- 3 دومينات بنفس الـ agent، وتبديل من الـ Sidebar.
- قراءة PDF (برقم الصفحة) وCSV وTXT.
- Index تدريجي بيتحمّل الـ rate limits.
- Evaluation شغّال على العينات (Medical 12/12 وPharma 10/10 وMarketing 10/10)، وكشف bug حقيقي واتصلح.
- **الـ Bonus (3 من 4):** واجهة شات، وذاكرة شخصية دائمة، وCalculator.
- Architecture diagram وDesign document (جاهزين في `docs/`).

**⏳ ناقص (مطلوب للتسليم)**
1. **مستندات حقيقية:** Medical جاهز تقريباً (NICE NG136 + NG28، الـ index وصل لـ 384 من 471 chunk، `python ingest.py medical` يكمّله). Pharma وMarketing لسه على العينات.
2. **Golden set حقيقي:** شغّل `python -m evaluation.question_forge --domain medical --n 10`، راجع الأسئلة في `evaluation/forged_medical.json` (امسح الغلط وصلّح الكلمات المفتاحية وضيف أسئلة بإيدك لحد 20-30)، وبعدها `python -m evaluation.run_eval --domain medical --file forged_medical.json` و`calibrate.py`.
3. **Demo:** فيديو 3-5 دقايق (سؤال عادي بمصادر، سؤال خطر يترفض، سؤال برة المجال، تبديل دومين).
4. **GitHub:** ارفع المشروع من غير `.env` و`chroma_db` والـ PDFs، وحط روابط المستندات في `knowledge_base/README.md`.
5. تحديث `docs/DESIGN.md` بنتايج التقييم الحقيقية.

**➖ مش معمول:** web search كأداة (الـ calculator كفاية لبونص الـ Tools).

## 6) ملاحظات مهمة
- **الـ quota المجاني:** الـ embeddings بتتحسب لكل chunk، والحد اليومي حوالي ألف. ابني دومين واحد في اليوم لو الملفات كبيرة. الـ ingest بيكمّل من حيث وقف، فمتخافش من تشغيله تاني.
- **الـ Index مرتبط بالـ embeddings:** لو غيّرت `GEMINI_EMBED_MODEL` لازم `python ingest.py <domain> --rebuild`.
- **التاسك طالب GPT / Azure OpenAI:** اخترنا Gemini عشان المجاني. اتكلم عن ده بصراحة في الـ interview: طبقة الموديل كلها في `llm.py`، فالانتقال لـ GPT تغيير ملف واحد.
- **الـ 100% في التقييم الحالي:** الأسئلة اتكتبت على العينات، فهي smoke test بس. التقييم الحقيقي بيجي مع المستندات الحقيقية.

## 7) الـ Bonus بالتفصيل

### الذاكرة الشخصية (Memory)
- في الـ Sidebar بتكتب اسمك وتختار دورك (clinician مثلاً) ومستوى التفصيل (brief/standard/detailed) واللغة (English/Arabic).
- الإعدادات وآخر 8 أسئلة بتتحفظ في `user_profiles/<اسمك>.json` وبترجع لما تفتح تاني.
- الـ agent بيستخدمها عشان يغيّر **الأسلوب والعمق واللغة بس**. قواعد الأمان مبتتغيّرش أبداً. فيه زرار "Forget my history" لمسح الذاكرة.

### الـ Calculator (ليه موجود؟)
الـ LLM بيغلط في الحساب. فلما تسأل "احسب BMI لـ 82 كيلو و175 سم" أو "ROI لو الإيراد 1500 والتكلفة 1000" الحساب بيتعمل في Python بدقة، والموديل بيفسّر النتيجة من المستندات بس. مفيش API ولا تكلفة إضافية (طلب LLM واحد زيادة للأسئلة اللي شكلها حساب بس). متضافش حسابات جرعات عمداً لأسباب أمان.

## 8) أسئلة متوقعة في الـ Interview
- **إزاي بتتعامل مع الـ hallucination؟** 7 طبقات: input guardrail، grounding gate، قواعد في الـ prompt، بوابة ثانية على ثقة الموديل، تحقق من الـ citations، confidence، disclaimer.
- **ليه chunk 900؟** توازن بين دقة الاسترجاع والسياق، والـ overlap 150 بيمنع قطع الفكرة.
- **إزاي بتضبط threshold؟** `calibrate.py` بيقيس أرقام التشابه لأسئلة صح وأسئلة برة المجال ويقترح قيمة بينهم.
- **إزاي تتوسّع؟** Azure AI Search/pgvector مع hybrid search وreranker، caching، multi-tenant، observability، تشغيل الـ golden set في CI.
- **أكبر نقطة ضعف؟** الـ regex guardrails ممكن تفوّت صياغات، فالحل في الإنتاج classifier أو moderation API.
