import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import RAGSystem
from src.utils import setup_logger

logger = setup_logger(__name__)

def main():
    rag_system = RAGSystem()

    file_path = input("Enter the path to your Excel file: ")
    try:
        rag_system.process_excel_file(file_path)
        logger.info("Excel file processed successfully.")
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        return

    while True:
        query = input("\nAsk a question (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        try:
            answer, sources = rag_system.answer_question(query)
            print(f"\nAnswer: {answer}")
            print("\nSources:")
            for source in sources[:3]:  # Limit to first 3 sources
                print(f"- {source[:100]}...")  # Print first 100 characters of each source
            
            while True:
                try:
                    rating = input("Please rate the answer from 1 (poor) to 5 (excellent), or press Enter to skip: ")
                    if rating == "":
                        break
                    rating = int(rating)
                    if 1 <= rating <= 5:
                        if rating < 3:
                            feedback = input("Please provide feedback on how the answer could be improved: ")
                            logger.info(f"Low rating feedback: {feedback}")
                        break
                    else:
                        print("Please enter a number between 1 and 5, or press Enter to skip.")
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 5, or press Enter to skip.")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()