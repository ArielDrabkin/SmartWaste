import json

# Define the class labels for the model's output
output_class = ["Plastic bottle/Can to deposit in Supermarkets", "Big Cardboard bin", "Unrecyclable garbage",
                "Glass - Purple bin", "Organic waste - Composter", "Grocery Packages - Orange bin",
                "Paper - Blue bin"]

# URLs for bin images
bin_images = ["https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Tomra_820.JPG/800px-Tomra_820.JPG",
              "https://www.greenkoala.co.il/wp-content/uploads/2017/08/WhatsApp-Image-2021-09-29-at-13.12.20.jpeg",
              "https://mediline.org.il/sites/mediline/cache/w_500,h_500,mode_pad/ofktpql3.jpg",
              "https://www.tmir.org.il/Download/Gallery/660-260-186_Pah_Sagol_pics4_14112021115647.jpg",
              "https://static.aujardin.info/cache/th/img9/composter-600x450.jpg",
              "https://zahalash.com/wp-content/uploads/2019/03/%D7%A2%D7%92%D7%9C%D7%AA-%D7%90%D7%A9%D7%A4%D7%94-360-%D7%9C%D7%99%D7%98%D7%A8-%D7%9B%D7%AA%D7%95%D7%9D.jpg",
              "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRn8eiuf5_6L6Lh5lDaFxwezeyZSVpk2PDOjYrH0cXDHDqihG5aBt3V7AlfvYSST3mdLvw&usqp=CAU"]

# Combine the lists into a dictionary
data_dict = {"output_class": output_class, "bin_images": bin_images}

# Write the dictionary to a JSON file
with open('data.json', 'w') as f:
    json.dump(data_dict, f)