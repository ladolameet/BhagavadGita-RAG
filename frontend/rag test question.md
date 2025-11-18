cd D:\rag_project\backend
cd D:\gita\backend


 uvicorn app:app --reload


Answer in a simple, easy-to-understand way.
Respond ONLY in plain text. 

# ✅ 1. Basic Understanding Questions (Check if model gets meaning right)

Ask things like:

 What is the Bhagavad Gita mainly about?
 Who is speaking in the Bhagavad Gita?
 Why was Arjuna confused?
 What is the central message of Krishna?

These test general understanding.

---

# ✅ 2. PDF-Specific Questions (Check if it retrieves correct text)

These help you see if FAISS is retrieving the correct pages & chunks.

Ask things like:

 How does the PDF define the Absolute Truth?
 What is bhakti-yoga according to this book?
 What is said about karma-yoga in this PDF?
 What does the author say about the soul (atma)?

Here, check the Sources shown in your frontend to confirm.

---

# ✅ 3. Chapter-Based Questions (Checks content recall)

Try:

 What happens in Chapter 2?
 What is described in Chapter 3?
 Explain Chapter 4 in simple words.
 What is the theme of Chapter 12?

This shows if chunking is okay.

---

# ✅ 4. Deep Meaning/Philosophy Questions (Tests summarization)

Try:

 What does Krishna say about surrender?
 What is the nature of the soul?
 What is dharma according to this book?
 Why does Krishna tell Arjuna to fight?

This tests reasoning + context blending.

---

# ✅ 5. Literal Questions (Check accuracy of extraction)

 What analogy is used to explain the soul?
 What example is given to describe material desires?
 How is yoga defined in the PDF?

 What is the birthdate of Arjuna?
 Did Krishna have a sister?
 What technology was used in Kurukshetra?



### Top 10 questions to test everything:

1. What is the main purpose of the Bhagavad Gita?
2. Why was Arjuna unwilling to fight?
3. Who is Krishna according to this PDF?
4. What is the Absolute Truth?
5. What is karma-yoga?
6. What is bhakti-yoga?
7. What does the Gita say about the soul?
8. What is the theme of Chapter 2?
9. Does the PDF mention reincarnation?
10. What is the easiest path to self-realization?



