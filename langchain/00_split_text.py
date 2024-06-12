from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Die vom KI Bundesverband gestartete Initiative LEAM – Large European AI Models – befasst sich mit dem Aufbau eines Hochleistungsrechenzentrums speziell für die Entwicklung von Künstlicher Intelligenz. Sogenannte KI-Foundation-Modelle werden sich disruptiv auf Wirtschaft und Gesellschaft auswirken. Derzeit wird dieser Paradigmenwechsel aber noch von in den USA und China entwickelten Modellen angeführt. 
Um Europas digitale Souveränität nicht zu gefährden und die Aufholjagd zu starten, hat der KI Bundesverband im Auftrag des Bundesministeriums für Wirtschaft und Klimaschutz, zusammen mit Vertretern aus Wirtschaft, Wissenschaft und Zivilgesellschaft, eine Machbarkeitsstudie lanciert. Diese zeigt auf, wie große KI Modelle auch in Deutschland entwickelt werden können.
Die Studie bestätigt, dass KI-Foundation-Modelle die Zukunft sind. Ihre Entwicklung wird in naher Zukunft viele neue Anwendungen, Plattformen und Geschäftsmodelle ermöglichen. LEAM zeigt einen Fahrplan auf, wie Deutschland an dieser Zukunftstechnologie teilhaben kann.
Im Zentrum von LEAM steht der Aufbau vertrauenswürdiger Open-Source Foundation-Modelle nach europäischen Ethikstandards. Außerdem wird Deutschlands Attraktivität als innovativer Wirtschaftsstandort und unsere internationale Wettbewerbsfähigkeit gesichert.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len
)

chunks = text_splitter.split_text(text)

for index, value in enumerate(chunks):
    print(f"Index: {index}, Value: {value}")
    print("-------------")