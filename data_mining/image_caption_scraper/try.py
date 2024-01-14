from image_caption_scraper import Image_Caption_Scraper

scraper = Image_Caption_Scraper(
                engine="all", # or "google", "yahoo", "flickr"
                num_images=100,
                query="dog chases cat",
                out_dir="images",
                headless=True,
                driver="chromedriver",
                expand=True,
                k=3
            )

scraper.scrape(save_images=True)