def draw_centroids(centroids, img, color=(255,0,0), radio=5):
    new_img = np.copy(img)
    for x, y in centroids:
        cv2.circle(new_img, (x, y), radio, color, -1)
    return new_img