import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64

model = load_model('artifacts/model.h5')
class_dict = np.load("artifacts/class_names.npy")
tab_icon = Image.open('artifacts/leaf_icon.png')
st.set_page_config(page_title='Ayurvedic Leaf Detection', page_icon=tab_icon)

def predict(image):
    IMG_SIZE = (1, 224, 224, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

contnt = "<p style='color:White; font-size: 16px; text-align:justify;'>Herbal medicine is the use of plants to treat disease and enhance general health and wellbeing. Herbs can interact with other pharmaceutical medications and should be taken with care. Herbal medicines are preferred in both developing and developed countries as an alternative to " \
         "synthetic drugs mainly because of no side effects. Recognition of these plants by human sight will be " \
         "tedious, time-consuming, and inaccurate.</p> " \
         "<p style='color:White; font-size: 16px; text-align:justify;'>Applications of image processing and computer vision " \
         "techniques for the identification of the medicinal plants are very crucial as many of them are under " \
         "extinction as per the IUCN records. Hence, the digitization of useful medicinal plants is crucial " \
         "for the conservation of biodiversity. So, Our project is used to identify 30 type of medicinal leafs using machine learning.</p>"
        
if __name__ == '__main__':
    add_bg_from_local("artifacts/Background.jpg")
    new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Ayurvedic Leaf Detection using Machine Learning</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(contnt, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        img = img.resize((300, 300))
        st.image(img)

        if st.button("Predict"):
            pred = predict(img)
            name = class_dict[pred]
            description = {0: "The rhizome of A. galanga is widely utilized in the preparation of Siddha and Ayurvedic preparations for several diseases including heart diseases, rheumatism, renal calculus, diabetes, hypertension, asthma, ulcer, bronchitis, inflammation, microbial infections, and chronic enteritis",
                           1: "Amaranthus viridis is used as traditional medicine in the treatment of fever, pain, asthma, diabetes, dysentery, urinary disorders, liver disorders, eye disorders and venereal diseases. The plant also possesses anti-microbial properties.",
                           2: "Jackfruit (Artocarpus heterophyllus Lam) is a rich source of several high-value compounds with potential beneficial physiological activities. It is well known for its antibacterial, antifungal, antidiabetic, anti-inflammatory, and antioxidant activities.",
                           3: "Neem leaf is rich in antioxidants and helps to boost the immune response in gum and tissues of the mouth. Neem offers a good remedy for curing mouth ulcers, tooth decay and acts as a pain reliever in toothache problems.",
                           4: "Basella alba is reported to improve testosterone levels in males, thus boosting libido. Decoction of the leaves is recommended as a safe laxative in pregnant women and children. Externally, the mucilaginous leaf is crushed and applied in urticaria, burns and scalds.",
                           5: "Mustard (Brassica juncea L.) is a Brassica vegetable that contains various health-promoting phytochemicals including carotenoids, phenolic compounds, and glucosinolates. The presence of sulfur-containing compounds such as glucosinolates in cruciferous vegetables such as Indian mustard reduces the risk of various types of cancer, including colon, kidney, and prostate cancer.",
                           6: "Its fruit is used in the ancient Indian herbal system of medicine, Ayurvedic, to treat acidity, indigestion, fresh and infected wounds, skin diseases, urinary disorders and diabetic ulcer, as well as biliousness, stomach pain, constipation, anemia, skin conditions, anorexia and insanity.",
                           7: "Lemons contain a high amount of vitamin C, soluble fiber, and plant compounds that give them a number of health benefits. Lemons may aid weight loss and reduce your risk of heart disease, anemia, kidney stones, digestive issues, and cancer.",
                           8: "Ficus racemosa Linn. (Moraceae) is a popular medicinal plant in India, which has long been used in Ayurveda, the ancient system of Indian medicine, for various diseases/disorders including diabetes, liver disorders, diarrhea, inflammatory conditions, hemorrhoids, respiratory, and urinary diseases.",
                           9: "It helps in reducing the elevated levels of liver enzymes and improving the liver cell degeneration, inflammation, and necrosis. This suggests that Ficus religiosa latex may have potential benefits in protecting the liver from drug-induced damage.",
                           10: "Hibiscus is high in antioxidants and offers many potential benefits. In particular, it may help promote weight loss, reduce the growth of bacteria and cancer cells, and support the health of the heart and liver. Hibiscus is available as an extract or, more often, a tea.",
                           11: "Jasmine aroma also helps in reducing anxiety, depression and stress. Ayurveda says that nervous system is controlled by Vata. Imbalance (depletion) of Vata causes weak memory or poor mental alertness. Jasmine helps manage mental alertness due to its Vata balancing and Medhya (brain tonic) properties.",
                           12: "Mango is one of the most popular of all tropical fruits. Mangiferin, being a polyphenolic antioxidant and a glucosyl xanthone, it has strong antioxidant, anti lipid peroxidation, immunomodulation, cardiotonic, hypotensive, wound healing, antidegenerative and antidiabetic activities.",
                           13: "Mint is a popular herb that may possess potential health benefits. This may include helping with digestive health, reducing allergic symptoms, and soothing common cold symptoms. Mint or mentha belongs to the Lamiaceae family, which contains around 15–20 plant species, including peppermint and spearmint.",
                           14: "Drumstick leaves are rich in calcium and phosphorus, which are crucial for bone health. Moringa leaves contain numerous antioxidants and nutrients, which improve the health and texture of our skin and hair. The hair pack of drumstick leaves reduces dandruff and gives shine and bounce to our hair.",
                           15: "Jamaica cherries are mainly consumed fresh, but the fruits and leaves are also used in traditional medicine to help relieve headaches and pain, reduce inflammation, and protect the overall health of the body. In addition to providing high nutritional properties to humans, the fruits are also consumed by bats and birds.",
                           16: "The Curry plant is well-known for the oil extracted from its flowers. The oil has medicinal properties that can heal burned skin or chapped lips. It serves as an anti-inflammatory and fungicidal astringent for skin.",
                           17: "Oleander has traditionally been used in the treatment of cardiac illness, asthma, diabetes mellitus, corns, scabies, cancer, and epilepsy, and in wound healing as an antibacterial/antimicrobial.",
                           18: "Nyctanthes arbor-tristis (Oleaceae) is a mythological plant; has high medicinal values in Ayurveda. The popular medicinal use of this plant are anti-helminthic and anti-pyretic besides its use as a laxative, in rheumatism, skin ailments and as a sedative.",
                           19: "Tulsi is believed to have antioxidant and anti-inflammatory properties that can help protect the liver from damage and support its overall health. Additionally, tulsi has been shown to help reduce stress and anxiety, which can have a positive impact on liver health. This plant is well known for its medicinal and spiritual properties in Ayurveda which includes aiding cough, asthma, diarrhea, fever, dysentery, arthritis, eye diseases, indigestion, gastric ailments, etc.",
                           20: "Betel leaves are used as a stimulant, an antiseptic, and a breath-freshener, whereas areca nut was considered as aphrodisiac. Chewing habits of people have changed over time. The betel leaves are chewed together in a wrapped package along with areca nut and mineral slaked lime.",
                           21: "Mexican Mint is also known as Indian Borage, Ajwain/Patharchur in Hindi, Ova Paan in Marathi. It has a plethora of health benefits that may include its ability to improve skin, detoxify the body, defend against cough and cold, ease arthritis pain, relieve stress, and optimize digestion.",
                           22: "Pongamia Pinnata Seed's have many industrial and medicinal uses. Seed oil has antiseptic, stimulant, and healing properties, and is applied in skin diseases, scabies, sores, herpes, and eczema. The seeds are said to have antidyslipidemic, anti-inflammatory, antiviral, antifilarial, and anti-ulcerogenic activities.",
                           23: "Psidium guajava (guava) is well known tropic tree grown in tropic areas for fruit. It is found to be effective in diarrhea, dysentery, gastroenteritis, hypertension, diabetes, caries, pain relief, cough, oral ulcers and to improve locomotors coordination and liver damage inflammation.",
                           24: "Pomegranate can be used in the prevention and treatment of several types of cancer, cardiovascular disease, osteoarthritis, rheumatoid arthritis, and other diseases. In addition, it improves wound healing and is beneficial to the reproductive system.",
                           25: "Sandalwood has antipyretic, antiseptic, antiscabetic, and diuretic properties. It is also effective in treatment of bronchitis, cystitis, dysuria, and diseases of the urinary tract. The main ingredient of sandalwood oil is α-santalol that has many therapeutic properties. It is used as a cosmetic to prevent wrinkles, reduce acne, dark spots, and heal wounds. It is also useful in skin conditions like psoriasis, eczema, Molluscum contagiosum, and scabies.",
                           26: "Jamun can give your immune system a significant boost. Vitamin C acts as an antioxidant and supports the production of white blood cells, enhancing the body's ability to fight infections and diseases. The presence of potassium in jamun makes it good for cardiovascular health.",
                           27: "The Syzygium jambos (Rose apple) fruits have wide range of medicinal properties to human being like control diabetes, reduce toxicity and boosts immune system etc. Plants develop massive root systems and can be useful for stabilizing soils on river banks.",
                           28: "Crape Jasmine is used to treat various diseases like rheumatic pain, headache, piles, inflammation, eye infections, abdominal tumors, strangury, arthralgia, epilepsy, fever, asthma, fractures, leprosy, paralysis, mania, oedema, rabies, skin diseases, urinary disorders, toothache, ulceration and vomiting. The broken stem exudes a toxic milky latex. However, in Ayurveda, plant parts are used in prescribed quantities to treat hypertension, headaches, scabies, and toothaches.",
                           29: "Fenugreek benefits may include managing blood sugar, alleviating menstrual cramps, and boosting breast milk supply. Cynthia Sass is a nutritionist and registered dietitian with master's degrees in both nutrition science and public health."}
            wiki_link = {0: "https://en.wikipedia.org/wiki/Alpinia_galanga",
                    1: "https://en.wikipedia.org/wiki/Amaranthus_viridis",
                    2: "https://en.wikipedia.org/wiki/Jackfruit",
                    3: "https://en.wikipedia.org/wiki/Azadirachta_indica",
                    4: "https://en.wikipedia.org/wiki/Basella_alba",
                    5: "https://en.wikipedia.org/wiki/Brassica_juncea",
                    6: "https://en.wikipedia.org/wiki/Carissa_carandas",
                    7: "https://en.wikipedia.org/wiki/Lemon",
                    8: "https://en.wikipedia.org/wiki/Ficus_auriculata",
                    9: "https://en.wikipedia.org/wiki/Ficus_religiosa",
                    10: "https://en.wikipedia.org/wiki/Hibiscus_rosa-sinensis",
                    11: "https://en.wikipedia.org/wiki/Jasmine",
                    12: "https://en.wikipedia.org/wiki/Mangifera_indica",
                    13: "https://en.wikipedia.org/wiki/Mentha",
                    14: "https://en.wikipedia.org/wiki/Moringa_oleifera",
                    15: "https://en.wikipedia.org/wiki/Muntingia",
                    16: "https://en.wikipedia.org/wiki/Curry_tree",
                    17: "https://en.wikipedia.org/wiki/Nerium",
                    18: "https://en.wikipedia.org/wiki/Nyctanthes_arbor-tristis",
                    19: "https://en.wikipedia.org/wiki/Ocimum_tenuiflorum",
                    20: "https://en.wikipedia.org/wiki/Betel",
                    21: "https://en.wikipedia.org/wiki/Coleus_amboinicus",
                    22: "https://en.wikipedia.org/wiki/Pongamia",
                    23: "https://en.wikipedia.org/wiki/Psidium_guajava",
                    24: "https://en.wikipedia.org/wiki/Pomegranate",
                    25: "https://en.wikipedia.org/wiki/Santalum_album",
                    26: "https://en.wikipedia.org/wiki/Syzygium_cumini",
                    27: "https://en.wikipedia.org/wiki/Syzygium_jambos",
                    28: "https://en.wikipedia.org/wiki/Tabernaemontana_divaricata",
                    29: "https://en.wikipedia.org/wiki/Fenugreek"}
            result = '<p style="font-family:sans-serif; color:White; font-size: 24px;">The given image ' \
                        'is <b>'+name+'</b></p>'
            benefit = ('<p style="font-family:sans-serif; color:White; font-size: 20px; text-align:justify;">Benefits of ' + name +' - '+description[pred]+'</p>')
            online_link = '<p style="font-family:sans-serif; color:White; font-size: 20px; text-align:justify;">To learn more about '+name+f', Please check here: <a href="{wiki_link[pred]}" style="color:White;">'+wiki_link[pred]+'</a></p>'
            st.markdown(result, unsafe_allow_html=True)
            st.markdown(benefit, unsafe_allow_html=True)
            st.markdown(online_link, unsafe_allow_html=True)
