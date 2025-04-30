import shutil

# Move the submission file to the correct folder (optional step)
shutil.move('submissions/titanic_predictions.csv', 'submissions/final_submission.csv')

print("Final submission file ready at 'submissions/final_submission.csv'")